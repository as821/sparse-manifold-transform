import torch
import math
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
        
        self.store_debug = False
        self.neighbor_indices = []
        self.neighbor_weighted_sim = []
        self.pred = []

    @torch.no_grad()
    def compute(self, chunk_size, train_set, sc_layer, smt_layer, num_train_images, test_features, test_targets):
        """Computes weighted k-NN accuracy @1 and @5. If cosine distance is selected,
        the weight is computed using the exponential of the temperature scaled cosine
        distance of the samples. If euclidean distance is selected, the weight corresponds
        to the inverse of the euclidean distance.

        Returns:
            Tuple[float]: k-NN accuracy @1 and @5.
        """
        # train_features = F.normalize(train_features)
        test_features = F.normalize(test_features).to("cuda")

        num_classes = torch.unique(test_targets).numel()
        num_test_images = test_targets.size(0)
        k = min(self.k, num_train_images)
        
        # calculate train set features then calculate the dot product and compute top-k neighbors
        mm_chnk = 250
        similarities = torch.zeros(size=(test_features.shape[0], num_train_images), dtype=test_features.dtype, device="cuda")
        train_targets = torch.zeros(size=(num_train_images,))
        pool = torch.nn.AvgPool2d(kernel_size=4, stride=2).to("cuda", non_blocking=True)
        for start in tqdm(range(0, num_train_images, mm_chnk)):
            end = min(start+mm_chnk, num_train_images)
            tf = torch.zeros(end - start, test_features.shape[1], device="cuda")
            for idx in range(start, end):
                embed, train_targets[idx] = train_set.generate_single_image_embedding(idx, sc_layer, smt_layer, cuda=True, train_img=True)
                
                # ImagePreprocessor.pool_single_image_patches, but without transfers
                sz = int(math.sqrt(embed.shape[0]))
                assert sz ** 2 == embed.shape[0]
                embed = pool(rearrange(embed, "(b c) a -> a b c", b=sz))
                embed /= (torch.linalg.vector_norm(embed, ord=2, dim=0, keepdim=True) + 1e-20)
                embed = embed.flatten()
                tf[idx - start] = F.normalize(embed, dim=0)
            similarities[:, start:end] = torch.mm(test_features, tf.T)

        # calculate k-NN from cosine similarities
        top1, top5, total = 0.0, 0.0, 0
        retrieval_one_hot = torch.zeros(k, num_classes, device="cuda")
        for idx in tqdm(range(0, num_test_images, chunk_size)):
            step_sz = min((idx + chunk_size), num_test_images) - idx
            targets = test_targets[idx : min((idx + chunk_size), num_test_images)]
            sim = similarities[idx : min(idx + chunk_size, num_test_images)]
            
            sim, indices = sim.topk(k, largest=True, sorted=True)
            indices = indices.cpu()
            candidates = train_targets.view(1, -1).expand(step_sz, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices).type(torch.int64)

            if torch.cuda.is_available():
                retrieved_neighbors = retrieved_neighbors.to('cuda', non_blocking=True)

            retrieval_one_hot.resize_(step_sz * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            sim = sim.clone().div_(self.T).exp_()


            probs = torch.sum(torch.mul(retrieval_one_hot.view(step_sz, -1, num_classes), sim.view(step_sz, -1, 1)), 1)
            _, predictions = probs.sort(1, True)
            predictions = predictions.cpu()
            
            if self.store_debug:
                self.neighbor_indices.append(indices)
                self.neighbor_weighted_sim.append(sim.cpu())
                self.pred.append(predictions)
            
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
    train_samples = len(train_set.dataset) if args.full_dset_eval else args.samples
    train_samples *= 2

    # train_embed, train_labels = train_set.generate_embeddings(train_samples, sc_layer, smt_layer, cuda=True, train_img=True)
    # train_embed = rearrange(train_embed, "a (b c) d -> a b c d", b=train_set.n_patch_per_dim)
    # train_embed = ImagePreprocessor.aggregate_image_embed(train_embed)

    # generate test set embeddings and labels
    test_set = generate_dset(args, 'test', train_set)
    test_embed, test_label = test_set.generate_embeddings(-1, sc_layer, smt_layer, cuda=True)
    test_embed = rearrange(test_embed, "a (b c) d -> a b c d", b=test_set.n_patch_per_dim)
    test_embed = ImagePreprocessor.aggregate_image_embed(test_embed)

    # Apply k-NN classifier to test set embeddings
    classifier = WeightedKNNClassifier(k=args.nnclass_k, T=args.knn_temp)
    return classifier.compute(args.classify_chunk, train_set, sc_layer, smt_layer, train_samples, test_embed, test_label)
