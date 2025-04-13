import torch
from tqdm import tqdm
import torch.nn.functional as F
from einops import rearrange

from preprocessor import ImagePreprocessor, generate_dset

import pdb


class WeightedKNNClassifier():
    # Taken from sololearn Github repo
    def __init__(self, k=20, T=0.07):
        """Implements the weighted k-NN classifier used for evaluation.
        Args:
            k (int, optional): number of neighbors. Defaults to 20.
            T (float, optional): temperature for the exponential. Only used with cosine
                distance. Defaults to 0.07.
        """
        self.k = k
        self.T = T
        self.output_probs = None

    @torch.no_grad()
    def compute(self, chunk_size, train_features, train_targets, test_features, test_targets):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        train_features = F.normalize(train_features)
        test_features = F.normalize(test_features)

        num_classes = torch.unique(test_targets).numel()
        num_train_images = train_targets.size(0)
        num_test_images = test_targets.size(0)
        num_train_images = train_targets.size(0)
        k = min(self.k, num_train_images)
        
        self.output_probs = torch.zeros((test_features.shape[0], num_classes))

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
            mm_chnk = 250
            similarities = torch.zeros(size=(features.shape[0], train_features.shape[1]), dtype=features.dtype, device=features.device)
            for start in range(0, train_features.shape[1], mm_chnk):
                end = min(start+mm_chnk, train_features.shape[1])
                tf = train_features[:, start:end]
                if torch.cuda.is_available():
                    tf = tf.to("cuda", non_blocking=True)   
                similarities[:, start:end] = torch.mm(features, tf)

            similarities, indices = similarities.topk(k, largest=True, sorted=True)
            indices = indices.cpu()
            candidates = train_targets.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices).type(torch.int64)

            if torch.cuda.is_available():
                retrieved_neighbors = retrieved_neighbors.to('cuda', non_blocking=True)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            similarities = similarities.clone().div_(self.T).exp_()

            probs = torch.sum(torch.mul(retrieval_one_hot.view(batch_size, -1, num_classes), similarities.view(batch_size, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            predictions = predictions.cpu()
            
            # find the predictions that match the target
            correct = predictions.eq(targets.data.view(-1, 1))

            # probs /= torch.sum(probs, dim=1, keepdim=True)
            # predicted_prob = probs[predictions[:, 0]]
            # correct_prob = probs[torch.arange(probs.shape[0]), targets.data.view(-1, 1)[:, 0].to(torch.int64)]
            # pdb.set_trace()

            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = (top5 + correct.narrow(1, 0, min(5, k, correct.size(-1))).sum().item())  # top5 does not make sense if k < 5
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5




def eval_knn_classifier(args, train_set, sc_layer, smt_layer):
    print("Test set evaluation.", flush=True)

    # generate train set embeddings + labels (include horizontal flip)
    train_samples = -1 if args.full_dset_eval else args.samples
    train_embed, train_labels = train_set.generate_embeddings(train_samples, sc_layer, smt_layer, cuda=True, train_img=True)
    train_embed = rearrange(train_embed, "a (b c) d -> a b c d", b=train_set.n_patch_per_dim)
    train_embed = ImagePreprocessor.aggregate_image_embed(train_embed)

    # generate test set embeddings and labels
    test_set = generate_dset(args, 'test', train_set)
    test_embed, test_label = test_set.generate_embeddings(-1, sc_layer, smt_layer, cuda=True)
    test_embed = rearrange(test_embed, "a (b c) d -> a b c d", b=test_set.n_patch_per_dim)
    test_embed = ImagePreprocessor.aggregate_image_embed(test_embed)

    # Apply k-NN classifier to test set embeddings
    classifier = WeightedKNNClassifier(k=args.nnclass_k, T=args.knn_temp)
    return classifier.compute(args.classify_chunk, train_embed, train_labels, test_embed, test_label)
