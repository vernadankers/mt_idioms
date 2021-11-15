import os
os.environ["TRANSFORMERS_OFFLINE"] = "1"
import logging
import argparse
import torch
import pickle
import numpy as np
import random
from transformers import MarianTokenizer, MarianMTModel


if __name__ == "__main__":
    random.seed(1)
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--folder", type=str, default="magpie/nl")
    parser.add_argument("--language", type=str, default="nl")
    args = vars(parser.parse_args())

    mname = f"Helsinki-NLP/opus-mt-en-{args['language']}"
    model = MarianMTModel.from_pretrained(mname, cache_dir="~/.cache/huggingface/transformers", local_files_only=True)
    tok = MarianTokenizer.from_pretrained(mname, cache_dir="~/.cache/huggingface/transformers", local_files_only=True)
    if torch.cuda.is_available():
        model = model.cuda()

    batch_size = 4
    TEST_FILE_SRC = args["source"]
    PRED_FILE = args["pred"]

    corpus_length = len(open(TEST_FILE_SRC).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}") 

    prds = dict()
    attention = dict()
    query_attention = dict()
    cross_attention = dict()
    hidden_states_enc = dict()
    hidden_states_dec = dict()

    with open(TEST_FILE_SRC, 'r', encoding="utf-8") as f_in, \
         open(os.path.join(f"{args['folder']}/prds", PRED_FILE), 'w', encoding="utf-8") as f_out:
        lines = f_in.readlines()
        for i in range(0, len(lines)+1, batch_size):
            if not lines[i:i+batch_size]:
                continue
            srcs, annotations = zip(*[x.split("\t")[:2] for x in lines[i:i+batch_size]])
            logging.info(f"Doing {i:4d}/{corpus_length:4d}")
            batch = tok.prepare_seq2seq_batch(
                src_texts=srcs, return_tensors="pt")
            for x in batch:
                if torch.cuda.is_available():
                    batch[x] = batch[x].cuda()

            # Gather the translations, attention and hidden states
            outputs_beam = model.generate(
                **batch, num_beams=5, max_length=512,
                return_dict_in_generate=True, output_hidden_states=True,
                output_attentions=True)
            # Rerun and collect the cross attention for the top prediction
            outputs_cross_attention = model.forward(
                **batch, return_dict=True, output_hidden_states=True,
                output_attentions=True, decoder_input_ids=outputs_beam["sequences"])

            # Put the batch size first
            attention_tmp = torch.stack(
                [x[0] for x in outputs_beam["encoder_attentions"]], dim=0).transpose(0, 1)
            query_attention_tmp = torch.stack(
                [x[1] for x in outputs_beam["encoder_attentions"]], dim=0).transpose(0, 1)
            hidden_states_enc_tmp = torch.stack(
                outputs_beam["encoder_hidden_states"], dim=0).transpose(0, 1)
            cross_attention_tmp = torch.stack(
                [x[0] for x in outputs_cross_attention["cross_attentions"]], dim=0).transpose(0, 1)

            detokenised_prds = tok.batch_decode(
                outputs_beam["sequences"], skip_special_tokens=True)

            for j, src in enumerate(srcs):
                f_out.write(f"{detokenised_prds[j]}\n")

                # Compute how the idiom annotation transfers to subtokens
                src_annotations = []
                for w, l in zip(src.split(), annotations[j].split()):
                    src_annotations.extend([int(l)] * len(tok.tokenize(w)))
                src_len = len(src_annotations) + 1

                # Collect attention in half-precision to reduce storage space
                prd = tok.convert_ids_to_tokens(outputs_beam["sequences"][j])[1:]
                attention[src] = np.array(attention_tmp[j][:, :, :src_len, :src_len].cpu().detach().numpy(), dtype="half")
                query_attention[src] = query_attention_tmp[j].cpu().detach().transpose(0, 1).numpy()[:src_len].tolist()
                hidden_states_enc[src] = hidden_states_enc_tmp[j].cpu().detach().transpose(0, 1).numpy()[:src_len].tolist()

                hidden_states_some_dropped = []
                query_states_some_dropped = []
                # For tokens that do not belong to the idiom, only store 10%
                for k, annotation in enumerate(src_annotations):
                    if k == 512:
                        break
                    if annotation == 1 or random.random() < 0.1:
                        hidden_states_some_dropped.append(hidden_states_enc[src][k])
                        query_states_some_dropped.append(query_attention[src][k])
                    else:
                        hidden_states_some_dropped.append(None)
                        query_states_some_dropped.append(None)
                hidden_states_enc[src] = hidden_states_some_dropped
                query_attention[src] = query_states_some_dropped
                cross_attention[src] = np.array(cross_attention_tmp[j][:, :, :, :src_len].cpu().detach().numpy(), dtype="half")
                prds[srcs[j]] = prd

    pickle.dump(dict(prds), open(
        f"{args['folder']}/tokenised_prds/{PRED_FILE.replace('.txt', '_prds.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(attention), open(
        f"{args['folder']}/attention/{PRED_FILE.replace('.txt', '_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(query_attention), open(
        f"{args['folder']}/query_attention/{PRED_FILE.replace('.txt', '_query_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(cross_attention), open(
        f"{args['folder']}/cross_attention/{PRED_FILE.replace('.txt', '_cross_attention.pickle')}", 'wb'), protocol=-1)
    pickle.dump(dict(hidden_states_enc), open(
        f"{args['folder']}/hidden_states_enc/{PRED_FILE.replace('.txt', '_hidden_states_enc.pickle')}", 'wb'), protocol=-1)
