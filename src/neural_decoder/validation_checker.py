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

unigram_counts = np.zeros(40)
bigram_counts = np.zeros((40,40))



def beam_search(model_output=None, beam_width=10, blank_id=0):

    # beams[prefix] = (p_blank, p_nonblank)
    beams = {(): (0.0, float('-inf'))}  # log-probs
    T, C = model_output.shape
    log_probs = torch.log_softmax(model_output, dim=-1)
    for t in range(T):
        new_beams = {}

        # Only consider top-k characters for speed
        topk_logp, topk_ids = torch.topk(log_probs[t], 40)
        for prefix, (pb, pnb) in beams.items():
            for logp, c in zip(topk_logp, topk_ids):
                c = int(c)
                candidates = []
                if c == blank_id:
                    # Extend with blank: always allowed
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
                        # Repeated char: 
                        new_pnb = torch.tensor(pnb) + logp
                        new_pb = float('-inf')
                        new_prefix = prefix
                        candidates.append((new_prefix, new_pb, new_pnb))

                        new_pnb2 = torch.tensor(pb) + logp
                        new_pb2 = float('-inf')
                        new_prefix2 = prefix + (c,)
                        candidates.append((new_prefix2, new_pb2, new_pnb2))
                    else:
                        # Normal extension
                        m = torch.logaddexp(torch.tensor(pb), torch.tensor(pnb))
                        new_pnb = m + logp
                        new_pb = float('-inf')
                        new_prefix = prefix + (c,)
                        candidates.append((new_prefix, new_pb, new_pnb))

                # Merge prefixes
                for nprefix, npb, npnb in candidates:
                    if nprefix not in new_beams:
                        new_beams[nprefix] = (float('-inf'), float('-inf'))

                    old_pb, old_pnb = new_beams[nprefix]
                    new_beams[nprefix] = (
                        torch.logaddexp(torch.tensor(old_pb), torch.tensor(npb)).item(),
                        torch.logaddexp(torch.tensor(old_pnb), torch.tensor(npnb)).item(),
                    )

        # prune to beam width
        beams = dict(sorted(
            new_beams.items(),
            key=lambda kv: torch.logaddexp(
                torch.tensor(kv[1][0]), torch.tensor(kv[1][1])
            ),
            reverse=True
        )[:beam_width])
        print(beams)

    # pick best prefix
    best_prefix = max(
        beams.items(),
        key=lambda kv: torch.logaddexp(torch.tensor(kv[1][0]), torch.tensor(kv[1][1]))
    )[0]

    return torch.tensor(best_prefix, dtype=torch.long)

# result = beam_search()
# print(result)
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
        for iterIdx in range(1):
            

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

    avgDayLoss = np.sum(allLoss) / len(testLoader)
    cer = total_edit_distance / total_seq_length
    print(avgDayLoss)
    print(cer)
