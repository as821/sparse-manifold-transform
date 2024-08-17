import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange

from preprocessor import ImagePreprocessor, generate_dset



class WeightedKNNClassifier():
    # Taken from sololearn Github repo
    def __init__(self, k=20, T=0.07, max_distance_matrix_size=int(5e6), distance_fx="cosine", epsilon=0.00001, dist_sync_on_step=False):
        """Implements the weighted k-NN classifier used for evaluation.
        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
            max_distance_matrix_size (int, optional): maximum number of elements in the
                distance matrix. Defaults to 5e6.
            distance_fx (str, optional): Distance function. Accepted arguments: "cosine" or
                "euclidean". Defaults to "cosine".
            epsilon (float, optional): Small value for numerical stability. Only used with
                euclidean distance. Defaults to 0.00001.
            dist_sync_on_step (bool, optional): whether to sync distributed values at every
                step. Defaults to False.
        """
        self.k = k
        self.T = T
        self.max_distance_matrix_size = max_distance_matrix_size
        self.distance_fx = distance_fx
        self.epsilon = epsilon

    @torch.no_grad()
    def compute(self, chunk_size, train_features, train_targets, test_features, test_targets):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        if self.distance_fx == "cosine":
            train_features = F.normalize(train_features)
            test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        k = min(self.k, num_train_images)

        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes).to(train_features.device)

        if(torch.cuda.is_available()):
            retrieval_one_hot = retrieval_one_hot.to("cuda", non_blocking=True)

        train_features = train_features.T
        for idx in tqdm(range(0, num_test_images, chunk_size)):
            # get the features for test images
            features = test_features[idx : min((idx + chunk_size), num_test_images), :]
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            batch_size = targets.size(0)

            if torch.cuda.is_available():
                features = features.to("cuda", non_blocking=True)

            # calculate the dot product and compute top-k neighbors
            if self.distance_fx == "cosine":
                mm_chnk = 250
                similarities = torch.zeros(size=(features.shape[0], train_features.shape[1]), dtype=features.dtype, device=features.device)
                for start in range(0, train_features.shape[1], mm_chnk):
                    end = min(start+mm_chnk, train_features.shape[1])
                    tf = train_features[:, start:end]
                    if torch.cuda.is_available():
                        tf = tf.to("cuda", non_blocking=True)   
                    similarities[:, start:end] = torch.mm(features, tf)

                # similarities = torch.mm(features, train_features)


            # elif self.distance_fx == "euclidean":
            #     similarities = 1 / (torch.cdist(features, train_features) + self.epsilon)
            else:
                raise NotImplementedError

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            indices = indices.cpu()
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices).type(torch.int64)

            if torch.cuda.is_available():
                retrieved_neighbors = retrieved_neighbors.to('cuda', non_blocking=True)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            if self.distance_fx == "cosine":
                similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), similarities.view(batch_size, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            predictions = predictions.cpu()

            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item())  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5




def test_set_classify(args, train_set, sc_layers, smt_layers, train_embed, train_labels):
    print("Test set evaluation.", flush=True)

    dset = generate_dset(args, 'test', train_set)
    betas, img_label = dset.generate_data(args.test_samples, test=True)

    if args.sc_only:
        if not isinstance(betas, (torch.Tensor)):
            betas = torch.from_numpy(betas)
        betas = sc_layers(args, betas, test=True).todense()
        betas = rearrange(betas, "d (a b c) -> a b c d", a=args.test_samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim)
    else:    
        # Apply pre-computed SMT to test set
        for idx, pr in enumerate(zip(sc_layers, smt_layers, args.embed_dim)):
            sc, smt, embed_dim = pr
            if idx != 0:
                betas = torch.from_numpy(betas).type(torch.float32)

            # Calculate embedding of test set images using learned SMT
            print("Calculating SMT embeddings...", flush=True)
            betas = sc(args, betas, test=True)
            betas = smt(betas)

        betas = rearrange(betas, "d (a b c) -> a b c d", a=args.test_samples, b=dset.n_patch_per_dim, c=dset.n_patch_per_dim, d=embed_dim)

    img_embed = ImagePreprocessor.aggregate_image_embed(betas)

    # Apply k-NN classifier to test set embeddings
    print("Calculating test accuracy...", flush=True)    
    return WeightedKNNClassifier(k=args.nnclass_k, T=args.knn_temp).compute(args.classify_chunk, train_embed, train_labels, img_embed, img_label)
