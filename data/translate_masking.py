import logging
import argparse
import pickle
import copy
from collections import defaultdict
import random
import torch
import numpy as np
from classifier import Classifier
from transformers import MarianTokenizer, MarianMTModel


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=str, default="magpie")
    parser.add_argument("--mode", type=str,
                        choices=["mask_regular", "mask_context", "mask_idiom"])
    parser.add_argument("--language", type=str, default="nl")
    args = vars(parser.parse_args())

    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{args['language']}.tsv")

    # Load the model and tokenizer, move t]o GPU if GPU available
    mname = f"Helsinki-NLP/opus-mt-en-{args['language']}"
    model = MarianMTModel.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()
    tok = MarianTokenizer.from_pretrained(mname)
    model.eval()

    for index in range(1727):
        logging.info(f"-----------------{index}-----------------")
        src_file = f"magpie/inputs/{index}.tsv"
        prd_file = f"{index}_pred.txt"

        corpus_length = len(open(src_file, encoding="utf-8").readlines())
        logging.info(f"Starting prediction... corpus size: {corpus_length}")
        hidden_states = defaultdict(list)
        attention = defaultdict(list)
        batch_size = 16

        with open(src_file, 'r', encoding="utf-8") as f_in:
            lines = f_in.readlines()
            indices_masked = defaultdict(list)

            for i in range(0, len(lines)+1, batch_size):
                for focus_layer in range(6):
                    if not lines[i:i + batch_size]:
                        continue
                    srcs, anns, idiom, _, _, pos = \
                        zip(*[x.split("\t") for x in lines[i:i + batch_size]])
                    logging.info(f"Doing {i:4d}/{corpus_length:4d}")

                    tok_src, extended_pos_tags, tok_annotations = [], [], []
                    for sentence, annotation, tags in zip(srcs, anns, pos):
                        tok_sent, tok_annotation, tok_tags = [], [], []
                        for w, l, t in zip(sentence.split(), annotation.split(), tags.split()):
                            tok_sent.extend(tok.tokenize(w))
                            tok_annotation.extend(
                                [int(l)] * len(tok.tokenize(w)))
                            tok_tags.extend([t] * len(tok.tokenize(w)))
                        tok_src.append(tok_sent)
                        extended_pos_tags.append(tok_tags)
                        tok_annotations.append(tok_annotation)

                    # Preprocess input data
                    batch = tok.prepare_seq2seq_batch(
                        src_texts=srcs,
                        tgt_texts=["target" for _ in srcs],
                        return_tensors="pt")
                    if torch.cuda.is_available():
                        for x in batch:
                            batch[x] = batch[x].cuda()

                    attention_mask = copy.deepcopy(batch["attention_mask"])
                    for j, src in enumerate(srcs):
                        if src not in indices_masked:
                            # Find an index to mask in the context
                            if args["mode"] == "mask_context":
                                indices = [
                                    k for k in range(len(tok_annotations[j]))
                                    # The index should not be in the idiom
                                    if tok_annotations[j][k] == 0
                                    # The index should be from a noun
                                    and extended_pos_tags[j][k] in ["NOUN"]
                                    # The index should have the idiom in its neighbourhood
                                    and 1 in tok_annotations[j][max(0, k - 10): k + 10 + 1]
                                    # The index should be from a token's first subtoken
                                    and u'▁' in tok_src[j][k]
                                    and k < 512]

                                # Now mask the index found (but mask only 1)
                                if indices:
                                    random.seed(1)
                                    idx = random.choice(indices)
                                    indices_masked[src] = idx

                            # Find an index to mask in the idiom
                            elif args["mode"] == "mask_idiom":
                                indices = [
                                    k for k in range(len(tok_annotations[j]))
                                    # The index should be from an idiom
                                    if tok_annotations[j][k] == 1
                                    # The index should be from a noun
                                    and extended_pos_tags[j][k] in ["NOUN"]
                                    # The index should be from a token's first subtoken
                                    and u'▁' in tok_src[j][k]
                                    and k < 512]

                                # Now mask the index found (but mask only 1)
                                if indices:
                                    random.seed(1)
                                    idx = random.choice(indices)
                                    indices_masked[src] = idx

                        idx = indices_masked[src]
                        attention_mask[j, idx] = 0

                    if torch.cuda.is_available():
                        attention_mask = attention_mask.cuda()

                    # Now run a forward pass with the model
                    outputs = model.forward(
                        **batch,
                        return_dict=True, output_hidden_states=True,
                        output_attentions=True, attention_from_layer=focus_layer,
                        custom_attention_mask=attention_mask)

                    # Extract attention & hidden states, put batch size first
                    attention_tmp = torch.stack(
                        [x[0] for x in outputs["encoder_attentions"]], dim=0).transpose(0, 1)
                    hidden_states_tmp = torch.stack(
                        outputs["encoder_hidden_states"], dim=0).transpose(0, 1)

                    for j, src in enumerate(srcs):
                        random.seed(1)
                        length = len(tok_annotations[j]) + 1
                        # focus_layer + 1 because 0 are the embeddings
                        vecs = hidden_states_tmp[j][focus_layer
                                                    + 1, :length, :].detach().cpu().numpy()
                        hidden_some_dropped = []
                        for k, annotation in enumerate(tok_annotations[j]):
                            if k == 512:
                                break
                            # Drop 10% for literal tokens to reduce amount of
                            # data stored to file
                            if annotation == 1 or random.random() < 0.1 or not classifier.contains(idiom):
                                hidden_some_dropped.append(vecs[k])
                            else:
                                hidden_some_dropped.append(None)
                        hidden_states[src].append(hidden_some_dropped)
                        vecs = attention_tmp[j][focus_layer,
                                                :, :length, :length]
                        attention[src].append(
                            np.array(vecs.detach().cpu().detach().numpy(), dtype="half"))

        all_encodings = {
            "hidden_states": dict(hidden_states),
            "attention": dict(attention),
            "indices_masked": dict(indices_masked)
        }

        pickle.dump(
            all_encodings,
            open(f"{args['folder']}/{args['mode']}/{prd_file.replace('txt', 'pickle')}", 'wb'))
