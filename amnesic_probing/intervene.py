import sys
sys.path.append('../data/')
sys.path.append("../probing")
from train_probes import set_seed
import random
import torch
from classifier import Classifier
from data import extract_sentences
import numpy as np
from tqdm import tqdm_notebook
import sacrebleu
import editdistance
from collections import defaultdict
import logging
logging.getLogger().setLevel(logging.INFO)
import pickle
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from debias import get_debiasing_projection



def generate(model, tokenizer, sentence, projection=False, attention_projection_indices=None,
             attention_projection_matrices=[None]*7, hidden_projection_indices=None,
             hidden_projection_matrices=[None]*7, gather_attention=False):
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[sentence], return_tensors="pt")
    if torch.cuda.is_available():
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()

    decoder_outputs = model.generate(
        **batch, beam_size=5, return_dict_in_generate=True,
        attention_projection_indices=attention_projection_indices,
        attention_projection_matrix=[x.cuda() if torch.cuda.is_available() and x is not None else x for x in attention_projection_matrices],
        hidden_projection_matrix=[x.cuda() if torch.cuda.is_available() and x is not None else x for x in hidden_projection_matrices],
        hidden_projection_indices=hidden_projection_indices,
        output_attentions=gather_attention)

    if gather_attention:
        attention = [x[0].cpu().detach().numpy() for x in decoder_outputs["encoder_attentions"]]
    else:
        attention = None

    tgt_tokens = tokenizer.convert_ids_to_tokens(decoder_outputs["sequences"][0])
    return "".join(tgt_tokens[1:]).replace("▁", " ").strip(), attention




def main(data, attention_projection_matrices, hidden_projection_matrices, baseline=False, filename="attention.pickle", attention_labels=[1], hidden_labels=[1], gather_attention=False):
    # Load attention weights
    model_version = 'Helsinki-NLP/opus-mt-en-nl'
    model = AutoModelForSeq2SeqLM.from_pretrained(model_version)
    tokenizer = AutoTokenizer.from_pretrained(model_version)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    classifier = Classifier(
        "../data/idiom_keywords_translated.tsv")

    equal_labels = []
    all_without = []
    all_with = []
    idiom_scores = dict()

    attention = defaultdict(list)
    for j in data:
        data[j] = [x for x in data[j] if x.magpie_label == "figurative" and x.translation_label == "paraphrase"]

    maxi = max(list(data.keys()))
    for j in sorted(list(data.keys())):
        if not data[j]:
            continue
        idiom = data[j][0].idiom
        idiom_equal_labels = []
        random.shuffle(data[j])

        for k, s in enumerate(data[j]):
            if (k + 1) % 20 == 0:
                logging.info(f"Sample {k} / {len(data[j])}")

            if s.magpie_label == "figurative" and s.translation_label == "paraphrase":
                without_projection, attention_without = generate(model, tokenizer, s.sentence, gather_attention=gather_attention)
                with_projection, attention_with = generate(
                    model, tokenizer, s.sentence, projection=True,
                    attention_projection_indices=s.index_select(attention_labels, no_subtokens=False),
                    attention_projection_matrices=attention_projection_matrices,
                    hidden_projection_indices=s.index_select(hidden_labels, no_subtokens=False),
                    hidden_projection_matrices=hidden_projection_matrices,
                    gather_attention=gather_attention)

                if gather_attention:
                    noun_pie_index = s.index_select(1, tags=["NOUN"])
                    noun_context_index = s.index_select(0, tags=["NOUN"], neighbours_only=True)
                    all_pie_index = s.index_select(1, tags=[])
                    all_context_index = s.index_select(0, tags=[], neighbours_only=True)

                old_label = classifier(s.idiom, without_projection)
                new_label = classifier(s.idiom, with_projection)
                if old_label == "paraphrase":
                    if new_label != "paraphrase":
                        print(s.sentence, without_projection, with_projection)
                    idiom_equal_labels.append(new_label)
                    equal_labels.append(new_label)
                    
                    if gather_attention:
                        attention[(old_label, new_label)].append((
                            attention_without, attention_with, noun_pie_index, noun_context_index,
                            all_pie_index, all_context_index, without_projection, with_projection))
 
                    all_with.append(with_projection)
                    all_without.append(without_projection)
        if not idiom_equal_labels:
            continue

        score = sacrebleu.corpus_bleu(all_with, [all_without]).score
        percentage_changed = 1 - (equal_labels.count("paraphrase") / len(equal_labels))
        idiom_scores[idiom] = 1 - (idiom_equal_labels.count('paraphrase') / len(idiom_equal_labels))
        logging.info(f"{idiom}: {idiom_scores[idiom]}")
        logging.info(f"Idiom {j}/{maxi}, {len(data[j])}: BLEU={score:.3f}, %={percentage_changed:.2f}, % idiom={np.mean(list(idiom_scores.values()))}"    )

    print(idiom_scores)

    #pickle.dump(dict(attention), open(filename, 'wb'))
    score = sacrebleu.corpus_bleu(all_with, [all_without]).score
    percentage_changed = (equal_labels.count("word-by-word") + equal_labels.count("copied")) / len(equal_labels)
    return percentage_changed, score, idiom_scores, attention


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--attention_layers", type=int, nargs='+', default=[])
    parser.add_argument("--hidden_labels", type=int, nargs='+', default=[1])
    parser.add_argument("--attention_labels", type=int, nargs="+", default=[1])
    parser.add_argument("--m", type=int, default=0)
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--filename", type=str, default="attention.pickle")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--num_classifiers", type=int, default=50)
    parser.add_argument("--gather_attention", action="store_true")
    args = parser.parse_args()

    set_seed(1)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.m, args.n, args.k):
        if (i + 1) % 50 == 0:
            logging.info(f"Sample {i/args.k:.0f} / {(args.n-args.m)/args.k:.0f}")
        samples = extract_sentences([i], use_tqdm=False)
        for s in samples:
            if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                s.label = 1
            elif s.translation_label == "word-by-word" and s.magpie_label == "literal":
                s.label = 0
            else:
                s.label = None

        samples = [s for s in samples if s.label is not None]
        if samples:
            data[i] = samples

    indices = list(data.keys())
    random.shuffle(indices)

    n = int(len(indices)/5)
    fold_1 = indices[:n] #[s for i in indices[:n] for s in data[i]]
    fold_2 = indices[n:n*2] #[s for i in indices[n:n*2] for s in data[i]]
    fold_3 = indices[n*2:n*3] #[s for i in indices[n*2:n*3] for s in data[i]]
    fold_4 = indices[n*3:n*4] #[s for i in indices[n*3:n*4] for s in data[i]]
    fold_5 = indices[n*4:] #[s for i in indices[n*4:] for s in data[i]]

    num_classifiers = 50
    input_dim = 512
    is_autoregressive = True

    attention_projection_matrices = []
    hidden_projection_matrices = []

    assert 0 not in args.attention_layers, "Cannot intervene om embs with attention queries!"

    percentages, bleus, idiom_percentages = [], [], []
    for fold in args.folds:
        if fold == 0:
            train = fold_3 + fold_4 + fold_5
            dev = fold_2
            test = fold_1
        elif fold == 1:
            train = fold_1 + fold_4 + fold_5
            dev = fold_3
            test = fold_2
        elif fold == 2:
            train = fold_1 + fold_2 + fold_5
            dev = fold_4
            test = fold_3
        elif fold == 3:
            train = fold_1 + fold_2 + fold_3
            dev = fold_5
            test = fold_4
        elif fold == 4:
            train = fold_2 + fold_3 + fold_4
            dev = fold_1
            test = fold_5

        for layer in range(7):
            if layer in args.hidden_layers:
                P = pickle.load(open(f"projection_matrices/hidden_fold={fold}_layer={layer}_baseline={args.baseline}_classifiers={args.num_classifiers}.pickle", 'rb'))
            else:
                P = None
            hidden_projection_matrices.append(P)
            if layer in args.attention_layers:
                P = pickle.load(open(f"projection_matrices/attention_fold={fold}_layer={layer}_baseline={args.baseline}_classifiers={args.num_classifiers}.pickle", 'rb'))
            else:
                P = None
            attention_projection_matrices.append(P)

        percentage, bleu, idiom_scores, attention = main(
            {i: data[i] for i in test}, attention_projection_matrices,
            hidden_projection_matrices, args.baseline, args.filename,
            args.attention_labels, args.hidden_labels, gather_attention=args.gather_attention)

        percentages.append(percentage)
        bleus.append(bleu)
        idiom_percentages.append(np.mean(list(idiom_scores.values())))
        pickle.dump(attention, open(f"attention_weights/attention_fold={fold}_layer={layer}_baseline={args.baseline}.pickle", 'wb'))

    logging.info(f"% = {np.mean(percentages):.2f}, BLEU = {np.mean(bleus):.3f}, % idioms = {np.mean(idiom_percentages)}")
