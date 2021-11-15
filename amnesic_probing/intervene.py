import sys
import random
import logging
import pickle
import sacrebleu
import numpy as np
import torch
from transformers import MarianTokenizer, MarianMTModel
sys.path.append('../data/')
from data import extract_sentences
from probing import set_seed
from classifier import Classifier
from collections import defaultdict

logging.getLogger().setLevel(logging.INFO)


def generate(model, tokenizer, sentence, projection=False, attention_projection_indices=None,
             attention_projection_matrices=[None]*7, hidden_projection_indices=None,
             hidden_projection_matrices=[None]*7, gather_attention=False):
    """
    Generate translations while projecting if projection is true.

    Args:
        - model
        - tokenizer
        - sentence (str)
        - projection (bool)
        - attention_projection_indices
        - attention_projection_matrices
        - hidden_projection_indices
        - hidden_projection_matrices
        - gather_attention (bool)

    Returns:
        - translation (str)
        - Attention matrix
    """
    batch = tokenizer.prepare_seq2seq_batch(src_texts=[sentence], return_tensors="pt")
    if torch.cuda.is_available():
        batch["input_ids"] = batch["input_ids"].cuda()
        batch["attention_mask"] = batch["attention_mask"].cuda()

    if torch.cuda.is_available():
        for i, matrix in enumerate(attention_projection_matrices):
            if matrix is not None:
                attention_projection_matrices[i] = matrix.cuda()
        for i, matrix in enumerate(hidden_projection_matrices):
            if matrix is not None:
                hidden_projection_matrices[i] = matrix.cuda()

    decoder_outputs = model.generate(
        **batch, num_beams=5, max_length=512, return_dict_in_generate=True,
        attention_projection_indices=attention_projection_indices,
        attention_projection_matrix=attention_projection_matrices,
        hidden_projection_matrix=hidden_projection_matrices,
        hidden_projection_indices=hidden_projection_indices,
        output_attentions=gather_attention)

    if gather_attention:
        attention = [att[0].cpu().detach().numpy()
                     for att in decoder_outputs["encoder_attentions"]]
    else:
        attention = None
    tgt_tokens = tokenizer.convert_ids_to_tokens(decoder_outputs["sequences"][0])
    return "".join(tgt_tokens[1:]).replace("▁", " ").strip(), attention


def main(language, data, attention_projection_matrices, hidden_projection_matrices,
         baseline=False, attention_labels=[1],
         hidden_labels=[1], gather_attention=False):
    """

    Args:
        - data
        - attention_projection_matrices
        - hidden_projection_matrices
        - baseline (bool)
        - attention_labels
        - hidden_labels
        - gather_attention

    Returns:

    """
    # Load the classifier
    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)
    # Load the model and tokenizer
    model = MarianMTModel.from_pretrained(mname)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()

    # Prune the data to only consider paraphrases
    all_without, all_with, attention = [], [], []
    changed_without, changed_with = [], []
    idiom_scores = dict()
    all_new_labels = []
    all_src = []

    maxi = max(list(data.keys()))
    for j in sorted(list(data.keys())):
        if not data[j]:
            continue
        idiom = data[j][0].idiom
        idiom_equal_labels = []

        for k, s in enumerate(data[j]):
            if (k + 1) % 20 == 0:
                logging.info(f"Sample {k} / {len(data[j])}")

            # Translate without the projection matrices
            without_projection, attention_without = generate(
                model, tokenizer, s.sentence, gather_attention=gather_attention)

            # Translate with the projection matrices
            with_projection, attention_with = generate(
                model, tokenizer, s.sentence, projection=True,
                attention_projection_indices=s.index_select(attention_labels, no_subtokens=False),
                attention_projection_matrices=attention_projection_matrices,
                hidden_projection_indices=s.index_select(hidden_labels, no_subtokens=False),
                hidden_projection_matrices=hidden_projection_matrices,
                gather_attention=gather_attention)

            # Collect the indices of PIE & context
            if gather_attention:
                noun_pie_index = s.index_select(1, tags=["NOUN"])
                noun_context_index = s.index_select(0, tags=["NOUN"], neighbours_only=True)
                all_pie_index = s.index_select(1, tags=[])
                all_context_index = s.index_select(0, tags=[], neighbours_only=True)

            old_label = classifier(s.idiom, without_projection)
            new_label = classifier(s.idiom, with_projection)
            if old_label == "paraphrase":
                idiom_equal_labels.append(new_label)
                all_new_labels.append(new_label)
                all_src.append(s)

                # Store attention
                if gather_attention:
                    attention.append((
                        attention_without, attention_with, noun_pie_index,
                        noun_context_index, all_pie_index, all_context_index))

                all_with.append(with_projection)
                all_without.append(without_projection)
                if new_label != "paraphrase":
                    changed_with.append(with_projection)
                    changed_without.append(without_projection)
        if not idiom_equal_labels:
            continue

        # Score this idiom with BLEU and the percentage of change
        idiom_scores[idiom] = 1 - (idiom_equal_labels.count('paraphrase') / len(idiom_equal_labels))
        logging.info(f"Idiom {j}/{maxi}, % = {idiom_scores[idiom]:.3f}"    )

    score1 = sacrebleu.corpus_bleu(all_with, [all_without]).score
    score2 = sacrebleu.corpus_bleu(changed_with, [changed_without]).score
    return score1, score2, idiom_scores, attention, {"src": all_src, "before": all_without, "after": all_with, "new_labels": all_new_labels}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_layers", type=int, nargs='+', default=[])
    parser.add_argument("--attention_layers", type=int, nargs='+', default=[])
    parser.add_argument("--hidden_labels", type=int, nargs='+', default=[1])
    parser.add_argument("--attention_labels", type=int, nargs="+", default=[1])
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stop", type=int, default=100)
    parser.add_argument("--step", type=int, default=1)
    parser.add_argument("--baseline", action="store_true")
    parser.add_argument("--filename", type=str, default="attention.pickle")
    parser.add_argument("--trace_filename", type=str, default="data/trace.pickle")
    parser.add_argument("--folds", type=int, nargs="+")
    parser.add_argument("--gather_attention", action="store_true")
    parser.add_argument("--language", type=str, default="nl")
    args = parser.parse_args()
    logging.info(vars(args))

    classifier = Classifier(
        f"../data/keywords/idiom_keywords_translated_{args.language}.tsv")
    mname = f"Helsinki-NLP/opus-mt-en-{args.language}"
    tokenizer = MarianTokenizer.from_pretrained(mname)

    set_seed(1)
    # Load all hidden representations of idioms
    data = dict()
    for i in range(args.start, args.stop, args.step):
        if (i + 1) % 50 == 0:
            logging.info(f"Sample {i/args.step:.0f} / {(args.stop-args.start)/args.step:.0f}")
        samples = extract_sentences([i], classifier, tokenizer, use_tqdm=False, data_folder=f"../data/magpie/{args.language}")
        for s in samples:
            if s.translation_label == "paraphrase" and s.magpie_label == "figurative":
                s.label = 1
            else:
                s.label = None
        if samples:
            idiom = samples[0].idiom
            samples = [s for s in samples if s.label is not None]
            data[idiom] = samples

    folds = pickle.load(open("../data/folds.pickle", 'rb'))
    fold_1 = folds[0]
    fold_2 = folds[1]
    fold_3 = folds[2]
    fold_4 = folds[3]
    fold_5 = folds[4]

    assert 0 not in args.attention_layers, "Cannot intervene om embs with attention queries!"

    bleus1, bleus2 = [], []
    all_idioms = dict()
    all_traces = defaultdict(list)
    for fold in args.folds:
        attention_projection_matrices = []
        hidden_projection_matrices = []

        if fold == 0:
            test = fold_1
        elif fold == 1:
            test = fold_2
        elif fold == 2:
            test = fold_3
        elif fold == 3:
            test = fold_4
        elif fold == 4:
            test = fold_5

        for layer in range(7):
            if layer in args.hidden_layers:
                P = pickle.load(
                    open(f"projection_matrices/{args.language}_hidden_fold={fold}_layer" +
                         f"={layer}_baseline={args.baseline}_classifiers=50.pickle", 'rb'))
            else:
                P = None
            hidden_projection_matrices.append(P)
            if layer in args.attention_layers:
                P = pickle.load(
                    open(f"projection_matrices/{args.language}_attention_fold={fold}_layer" +
                         f"={layer}_baseline={args.baseline}_classifiers=50.pickle", 'rb'))
            else:
                P = None
            attention_projection_matrices.append(P)

        bleu1, bleu2, idiom_scores, attention, trace = main(args.language,
            {i: data[i] for i in test}, attention_projection_matrices,
            hidden_projection_matrices, args.baseline, args.attention_labels,
            args.hidden_labels, args.gather_attention)

        for x in idiom_scores:
            all_idioms[x] = idiom_scores[x]

        bleus1.append(bleu1)
        bleus2.append(bleu2)
        for x in trace:
            all_traces[x] = all_traces[x] + trace[x]
        if args.gather_attention:
            pickle.dump(attention, open(args.filename, 'wb'))

        pickle.dump((all_traces, all_idioms), open(args.trace_filename, 'wb'))
        percentage = np.mean(list(all_idioms.values()))
        percentage2 = 1 - (all_traces["new_labels"].count('paraphrase') / len(all_traces["new_labels"]))
        logging.info(f"BLEU1 = {np.mean(bleus1):.3f}, BLEU2 = {np.mean(bleus2):.3f}, % idioms = {percentage}, % idioms = {percentage2}")
        print(all_idioms)
