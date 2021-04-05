import logging
import argparse
import os
import torch
import pickle
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer


def main(args):
    mname = f'Helsinki-NLP/opus-mt-en-nl'
    model = AutoModelForSeq2SeqLM.from_pretrained(mname)
    tok = AutoTokenizer.from_pretrained(mname)

    batch_size = 32
    corpus_length = len(open(args["source"]).readlines())
    logging.info(f"Starting prediction... corpus size: {corpus_length}") 

    prds = dict()
    attention = dict()
    cross_attention = dict()
    hidden_states_enc = dict()
    hidden_states_dec = dict()

    with open(args["source"], 'r', encoding="utf-8") as f_in, \
         open(os.path.join(f"{args['folder']}/prds", args["pred"]), 'w', encoding="utf-8") as f_out:
        lines = f_in.readlines()
        for i in range(0, len(lines) + 1, batch_size):
            if not lines[i:i + batch_size]:
                continue
            srcs, annotations = zip(*[x.split("\t")[:2] for x in lines[i:i + batch_size]])
            logging.info(f"Doing {i:4d}/{corpus_length:4d}")
            batch = tok.prepare_seq2seq_batch(
                src_texts=srcs, return_tensors="pt")

            # Get the model's predicted translations with beam search, size 5
            outputs_beam = model.generate(
                **batch, num_beams=5, max_length=512,
                return_dict_in_generate=True, output_hidden_states=True,
                output_attentions=True)

            # Now feed the predicted translation through the model to get the
            # cross-attention
            outputs_cross_attention = model.forward(
                **batch, return_dict=True, output_hidden_states=True,
                output_attentions=True, decoder_input_ids=outputs_beam["sequences"])

            # Transposes to get batch-first setups
            attention_tmp = torch.stack(
                outputs_beam["encoder_attentions"], dim=0).transpose(0, 1)
            hidden_states_enc_tmp = torch.stack(
                outputs_beam["encoder_hidden_states"], dim=0).transpose(0, 1)
            cross_attention_tmp = torch.stack(
                outputs_cross_attention["cross_attentions"], dim=0).transpose(0, 1)
            hidden_states_dec_tmp = torch.stack(
                outputs_cross_attention["decoder_hidden_states"], dim=0).transpose(0, 1)
            detokenised_prds = tok.batch_decode(
                outputs_beam["sequences"], skip_special_tokens=True)

            for j, src in enumerate(srcs):
                # Write the actual translation, detokenised, to file
                f_out.write(f"{detokenised_prds[j]}\n")

                # Get the tokenised length of each input
                src_annotations = []
                for w, l in zip(src.split(), annotations[j].split()):
                    src_annotations.extend([l] * len(tok.tokenize(w)))
                src_len = len(src_annotations) + 1

                # Extract the relevant parts of the attention / hidden states
                prd = tok.convert_ids_to_tokens(outputs_beam["sequences"][j])[1:]
                attention[src] = attention_tmp[j][:, :, :src_len, :src_len].detach().numpy()
                hidden_states_enc[src] = hidden_states_enc_tmp[j].detach().numpy()
                cross_attention[src] = cross_attention_tmp[j][:, :, :, :src_len].detach().numpy()
                hidden_states_dec[src] = hidden_states_dec_tmp[j].detach().numpy()

                # Also store the tokenised translation, to file
                prds[srcs[j]] = prd

    # Store hidden states / attention in separate folders
    fn = f"{args['folder']}/tokenised_prds/{args["pred"].replace('.txt', '_prds.pickle')}"
    pickle.dump(dict(prds), open(fn, 'wb'), protocol=-1)
    fn = f"{args['folder']}/attention/{args["pred"].replace('.txt', '_attention.pickle')}"
    pickle.dump(dict(attention), open(fn, 'wb'), protocol=-1)
    fn = f"{args['folder']}/cross_attention/{args["pred"].replace('.txt', '_cross_attention.pickle')}"
    pickle.dump(dict(cross_attention), open(fn, 'wb'), protocol=-1)
    fn = f"{args['folder']}/hidden_states_enc/{args["pred"].replace('.txt', '_hidden_states_enc.pickle')}"
    pickle.dump(dict(hidden_states_enc), open(fn, 'wb'), protocol=-1)
    fn = f"{args['folder']}/hidden_states_dec/{args["pred"].replace('.txt', '_hidden_states_dec.pickle')}"
    pickle.dump(dict(hidden_states_dec), open(fn, 'wb'), protocol=-1)


if __name__ == "__main__":
    logging.getLogger().setLevel(logging.INFO)
    logging.basicConfig(format='%(asctime)s - %(message)s',
                        datefmt='%d-%b %H:%M:%S')
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--folder", type=str, default="per_idiom_punct")
    args = vars(parser.parse_args())

    main(args)
