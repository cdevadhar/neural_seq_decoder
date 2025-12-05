import torch
from torch import nn
from .neural_decoder_trainer import loadModel
from .neural_decoder_trainer import getDatasetLoaders
import numpy as np
from .label_smoothing_ctc import LabelSmoothingCTCLoss
from edit_distance import SequenceMatcher
import warnings
warnings.filterwarnings("ignore")

model, args = loadModel("/home/cdevadhar/neural_seq_decoder/baseline_speech_logs/speechBaseline4")

trainLoader, testLoader, loadedData = getDatasetLoaders(
        args["datasetPath"],
        args["batchSize"],
    )
device = "cuda"
loss_ctc = LabelSmoothingCTCLoss()

unigram_counts = np.zeros(41)
bigram_counts = np.zeros((41,41))

for _, y, _, y_len, dayIdx in trainLoader:
    y = y.cpu().numpy()
    y_len = y_len.cpu().numpy()
    for seq, seq_len in zip(y, y_len):
        seq = seq[0:seq_len]
        if len(seq) == 0:
            continue

        for tok in seq:
            unigram_counts[tok] += 1
        for i in range(len(seq) - 1):
            bigram_counts[seq[i], seq[i+1]] += 1


epsilon = 1e-3  

unigram_probs = (unigram_counts + epsilon) / (unigram_counts.sum() + epsilon * 41)
unigram_log_probs = np.log(unigram_probs)

bigram_probs = bigram_counts + epsilon
row_sums = bigram_probs.sum(axis=1, keepdims=True)
bigram_probs = bigram_probs / row_sums
bigram_log_probs = np.log(bigram_probs)

print(unigram_log_probs)
print(bigram_log_probs)

def beam_search(model_output=None, beam_width=10, blank_id=0, bigram_weight=0):

    # beams[prefix] = (p_blank, p_nonblank)
    beams = {(): (0.0, float('-inf'))}  # log-probs
    T, C = model_output.shape
    log_probs = torch.log_softmax(model_output, dim=-1)
    for t in range(T):
        new_beams = {}

        topk_logp, topk_ids = torch.topk(log_probs[t], 10)
        for prefix, (pb, pnb) in beams.items():
            for logp, c in zip(topk_logp, topk_ids):
                c = int(c)
                candidates = []
                if c == blank_id:
                    new_pb = torch.logaddexp(
                        torch.tensor(pb),
                        torch.tensor(pnb)
                    ) + logp
                    new_pnb = float('-inf')
                    new_prefix = prefix
                    candidates.append((new_prefix, new_pb, new_pnb))
                else:
                    # Extend with the same character
                    if len(prefix) > 0 and prefix[-1] == c:
                        new_pnb = torch.tensor(pnb) + logp
                        new_pb = float('-inf')
                        new_prefix = prefix
                        candidates.append((new_prefix, new_pb, new_pnb))
                        
                        prev = prefix[-1]
                        ngram_prob = bigram_log_probs[prev, c]
                        new_pnb2 = torch.tensor(pb) + logp + bigram_weight*ngram_prob
                        new_pb2 = float('-inf')
                        new_prefix2 = prefix + (c,)
                        candidates.append((new_prefix2, new_pb2, new_pnb2))
                    else:
                        # Normal extension
                        # apply ngram again because ur extending
                        ngram_prob  = 0
                        if len(prefix) > 0:
                            prev = prefix[-1]
                            ngram_prob = bigram_log_probs[prev, c]
                        else:
                            ngram_prob = unigram_log_probs[c]
                        # print(ngram_prob)

                        m = torch.logaddexp(torch.tensor(pb), torch.tensor(pnb))
                        new_pnb = m + logp + bigram_weight*ngram_prob
                        new_pb = float('-inf')
                        new_prefix = prefix + (c,)
                        candidates.append((new_prefix, new_pb, new_pnb))

                # Merge shit
                for nprefix, npb, npnb in candidates:
                    if nprefix not in new_beams:
                        new_beams[nprefix] = (float('-inf'), float('-inf'))
                    old_pb, old_pnb = new_beams[nprefix]
                    new_beams[nprefix] = (
                        torch.logaddexp(torch.tensor(old_pb), torch.tensor(npb)).item(),
                        torch.logaddexp(torch.tensor(old_pnb), torch.tensor(npnb)).item(),
                    )

        beams = dict(sorted(
            new_beams.items(),
            key=lambda kv: torch.logaddexp(
                torch.tensor(kv[1][0]), torch.tensor(kv[1][1])
            ),
            reverse=True
        )[:beam_width])

    best_prefix = max(
        beams.items(),
        key=lambda kv: torch.logaddexp(torch.tensor(kv[1][0]), torch.tensor(kv[1][1]))
    )[0]

    return torch.tensor(best_prefix, dtype=torch.long)


with torch.no_grad():
    model.eval()
    allLoss = []
    total_edit_distance = 0
    total_edit_distance_beam = 0
    total_seq_length = 0
    for X, y, X_len, y_len, testDayIdx in testLoader:
        X, y, X_len, y_len, testDayIdx = (
            X.to(device),
            y.to(device),
            X_len.to(device),
            y_len.to(device),
            testDayIdx.to(device),
        )

        pred = model.forward(X, testDayIdx)
        loss = loss_ctc(
            torch.permute(pred.log_softmax(2), [1, 0, 2]),
            y,
            ((X_len - model.kernelLen) / model.strideLen).to(torch.int32),
            y_len,
        )
        loss = torch.sum(loss)
        allLoss.append(loss.cpu().detach().numpy())

        adjustedLens = ((X_len - model.kernelLen) / model.strideLen).to(
            torch.int32
        )
        print(pred.shape[0])
        for iterIdx in range(pred.shape[0]):

            decodedSeq = beam_search(
                torch.tensor(pred[iterIdx, 0 : adjustedLens[iterIdx], :].cpu())
            )

            # decodedSeq = torch.unique_consecutive(decodedSeq, dim=-1)
            decodedSeq = decodedSeq.cpu().detach().numpy()
            decodedSeq = np.array([i for i in decodedSeq if i != 0])

            trueSeq = np.array(
                y[iterIdx][0 : y_len[iterIdx]].cpu().detach()
            )
            matcher = SequenceMatcher(
                a=trueSeq.tolist(), b=decodedSeq.tolist()
            )
            total_edit_distance += matcher.distance()

            total_seq_length += len(trueSeq)
            cer = total_edit_distance / total_seq_length
            print("CURRENT ONE")
            print(matcher.distance() / len(trueSeq))
            print("CUMULATIVE")
            print(cer)

    avgDayLoss = np.sum(allLoss) / len(testLoader)
    cer = total_edit_distance / total_seq_length
    print(avgDayLoss)
    print(cer)
