# mt_idioms

### Install

1 - Install libraries using `requirements.txt`.

2 - Download a Spacy model for English using `python -m spacy download en_core_web_sm`.

3 - Custom install of the `transformers` library:
`transformers==4.3.0`, with `modeling_marian.py` replaced with the adapted version from this repository.
To clone `transformers` and replace MarianMT with our adapted version, carry out the following instructions:
```
git clone https://github.com/huggingface/transformers.git
cd transformers
git checkout 800f385d7808262946987ce91c158186649ec954
cd ..
mv modeling_marian.py transformers/src/transformers/models/marian/modeling_marian.py
cd transformers
pip install -e .
```

Other non-standard libraries are provided in the `requirements.txt` file.

### Experiments
For results from the paper's sections, visit the following folders, that have their own README.
- Section 3: See the `data` folder.
- Section 4: See the `attention` folder.
- Section 5: See the `hidden_representations` folder.
- Section 6: See the `amnesic_probing` folder.
- Appendix B: See the `data` folder.
- Appendix C: See the `attention` folder.
- Appendix D: See the `attention` and `hidden_representations` folder.
- Appendix E: See the `hidden_representations` folder.
- Appendix F: See the `amnesic_probing` folder.
- Appendix G: See the `data` folder.

<image src="attention_flow.png" />
