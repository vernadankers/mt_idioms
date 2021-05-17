import logging
import argparse
import pickle
from collections import defaultdict
import random
import torch

from transformers import MarianTokenizer, MarianMTModel


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--folder", type=str, default="magpie")
    parser.add_argument("--mode", type=str,
                        choices=["mask_regular", "mask_context", "mask_idiom"])
    args = vars(parser.parse_args())

    # Load the model and tokenizer, move to GPU if GPU available
    mname = f'Helsinki-NLP/opus-mt-en-nl'
    model = MarianMTModel.from_pretrained(mname)
    if torch.cuda.is_available():
        model = model.cuda()
    tok = MarianTokenizer.from_pretrained(mname)
    model.eval()

    corpus_length = len(open(args["source"]).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}")

    hidden_states = defaultdict(list)
    attention = defaultdict(list)
    batch_size = 16

    with open(args["source"], 'r', encoding="utf-8") as f_in:
        lines = f_in.readlines()

        for i in range(0, len(lines)+1, batch_size):
            for focus_layer in range(6):
                if not lines[i:i + batch_size]:
                    continue
                srcs, _, _, _, _, _, tok_src, tok_annotations, extended_pos_tags = \
                    zip(*[x.split("\t") for x in lines[i:i + batch_size]])
                logging.info(f"Doing {i:4d}/{corpus_length:4d}")
                tok_src = [l.split() for l in tok_src]
                extended_pos_tags = [l.split() for l in extended_pos_tags]
                tok_annotations = [[int(n) for n in l.split()] for l in tok_annotations]

                # Preprocess input data
                batch = tok.prepare_seq2seq_batch(
                    src_texts=srcs,
                    #dummy text, we are only interested in the encoder's outputs
                    tgt_texts=["target" for _ in srcs],
                    return_tensors="pt")
                if torch.cuda.is_available():
                    for x in batch:
                        batch[x] = batch[x].cuda()

                attention_mask = torch.zeros(batch["input_ids"].shape)
                for j, src in enumerate(srcs):
                    # Fill the mask with 1s for the sentence + </s> token
                    for k in range(len(tok_annotations[j]) + 1):
                        attention_mask[j, k] = 1

                    # Find an index to mask in the context
                    if args["mode"] == "mask_context":
                        indices = [
                            k for k in range(len(tok_annotations[j]))
                            # The index should not be in the idiom
                            if tok_annotations[j][k] == 0
                            # The index should be from a noun
                            and extended_pos_tags[k] in ["NOUN"]
                            # The index should have the idiom in its neighbourhood
                            and 1 in tok_annotations[j][k - 10: k + 10 + 1]
                            # The index should be from a token's first subtoken
                            and u'▁' in tok_src[j][k]]

                        # Now mask the index found (but mask only 1)
                        if indices:
                            attention_mask[j, random.choice(indices)] = 0

                    # Find an index to mask in the idiom
                    elif args["mode"] == "mask_idiom":
                        indices = [
                            k for k in range(len(tok_annotations[j]))
                            # The index should be from an idiom
                            if tok_annotations[j][k] == 1
                            # The index should be from a noun
                            and extended_pos_tags[j][k] in ["NOUN"]
                            # The index should be from a token's first subtoken
                            and u'▁' in tok_src[j][k]]

                        # Now mask the index found (but mask only 1)
                        if indices:
                            attention_mask[j, random.choice(indices)] = 0

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
                    outputs["encoder_attentions"], dim=0).transpose(0, 1)
                hidden_states_tmp = torch.stack(
                    outputs["encoder_hidden_states"], dim=0).transpose(0, 1)

                for j, src in enumerate(srcs):
                    length = len(tok_annotations[j]) + 1

                    # focus_layer + 1 because 0 are the embeddings
                    vecs = hidden_states_tmp[j][focus_layer + 1, :length, :]
                    hidden_states[src].append(vecs.detach().cpu().numpy())
                    vecs = attention_tmp[j][focus_layer, :, :length, :length]
                    attention[src].append(vecs.detach().cpu().numpy())

    all_encodings = {
        "hidden_states": dict(hidden_states),
        "attention": dict(attention)
    }

    pickle.dump(
        all_encodings,
        open(f"{args['folder']}/{args['mode']}/{args['pred'].replace('txt', 'pickle')}", 'wb'))
