# Workflow

All scripts resolve their paths from this folder (the project root) and write to `make_datasets/output/`, `outputs/` and the subfolders named in each script. The header of each script describes its inputs, outputs and options.

Output files carry the date and time of the run in their names. The file names written in the scripts (for example `scenario_desc_gen_gpt-5.6_20260826_0429.xlsx`) identify the runs reported in the paper. After a step is run again, its new output is passed to the later steps through the environment variables listed in the script headers, or by updating the file name in the script.

1. Dataset: `python3 make_datasets/build_r10_sd_mixed.py`
2. Scenario descriptions: `python3 scenario_des/scenario_desc_generate.py`, then `python3 scenario_des/scenario_desc_apply.py`. `python3 scenario_des/scenario_desc_repeat_eval.py` compares repeated generation runs
3. Model fingerprints: `python3 fingerprint/fingerprint_from_papers_comp_evidence.py`. `python3 fingerprint/fp_compare_bert.py FILE1.xlsx FILE2.xlsx` compares repeated extraction runs
4. Design dimensions and coverage, from `policy_coverage_analysis/`: `python3 coverage/classify.py --model gpt-5.6`, then `python3 coverage/report.py`
5. Training, validation and test sets: `python3 make_datasets/split_r10_sd.py`
6. Zero-shot RAG baseline and ablations: `python3 rag_core.py` and the three ablation scripts
7. Finetuning: `python3 finetune/prepare_sft_data.py`, then `finetune/colab_finetune_qwen3.5-9b.ipynb` and `finetune/colab_infer_vllm.ipynb`
8. Baseline models: `python3 knn/knn_regression.py --condition 1`, `python3 cvae/train_cvae.py --condition 1 --sweep`, then `python3 cvae/sweep_val_preds.py --condition 1`
9. Climate category classifiers: `python3 classifier/train_c_classifier.py`, then `python3 classifier/apply_c_classifier.py`
10. Analyses for the figures: `python3 neighbor_stats_val.py`, `python3 scenario_embedding_pair_check.py` and `python3 sparse_test_analysis.py`
11. Figures, from `results/`: `python3 build_main_data.py` and `python3 build_si_data.py` compute the source data, and `python3 make_main_figs.py` and `python3 make_si_figs.py` draw the figures

The two plotting scripts need only the two workbooks in `results/`, so the figures can be redrawn without the other steps.
