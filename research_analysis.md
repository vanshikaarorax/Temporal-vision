# 7. Research Analysis

This section interprets the experiments as evidence about when temporal information is useful, rather than treating the highest accuracy as the only objective. The conclusions below are limited to the nine-class HMDB51 subset and the experiments implemented in this project.

## A. When did temporal information help?

Temporal information helped most clearly with **CLIP embeddings**. The CLIP Transformer achieved the strongest result in the single-frame-versus-temporal comparison, with **98.60% accuracy and 0.9873 macro-F1**. This was higher than the CLIP single-frame MLP (**94.41% accuracy, 0.9479 macro-F1**) and the CLIP mean-pooling MLP (**97.20% accuracy, 0.9753 macro-F1**).

This suggests that, for this encoder and dataset split, a learned temporal model could use information across multiple frames more effectively than relying on one middle frame or simply averaging the frame embeddings. CLIP mean pooling also improved over the single-frame baseline, indicating that access to multiple observations was useful even without an order-aware model.

The result is encoder-dependent: the experiments do not show that temporal modeling is uniformly beneficial for every visual representation.

## B. When did it not help?

Temporal modeling did **not** consistently improve performance with SigLIP embeddings. The SigLIP single-frame MLP achieved **97.20% accuracy and 0.9746 macro-F1**, outperforming the SigLIP mean-pooling MLP (**95.10%, 0.9555 macro-F1**), GRU (**93.71%, 0.9443**), LSTM (**93.71%, 0.9379**), Transformer (**95.10%, 0.9561**) and Temporal CNN (**95.80%, 0.9621**).

Therefore, for SigLIP in this experiment, adding a temporal architecture did not translate into better classification. A strong single frame was already highly informative for these classes, and the additional sequence processing may not have added useful discriminative information.

There is also an important distinction between **using multiple frames** and **using their temporal order**. In the order-sensitivity experiment, the CLIP Transformer scored 0.9873 macro-F1 on chronological sequences, 0.9811 on reversed sequences and a shuffled mean of 0.9805 (standard deviation 0.0052). The chronological advantage over the shuffled mean was only **0.0068 macro-F1**, or **0.68 percentage points**. For the SigLIP Transformer, the corresponding scores were 0.9561 chronological, 0.9496 reversed and 0.9534 shuffled mean (standard deviation 0.0034).

These results show some sensitivity to sequence ordering, but the differences are modest. They do not establish that the models rely strongly on the correct temporal relationships. In particular, a model can benefit from seeing several frames while still depending mainly on the additional visual evidence rather than on motion order.

## C. How can you determine whether a temporal model is actually using temporal relationships rather than simply receiving more visual information?

The key is to control the **number and identity of observed frames** while changing only their temporal arrangement.

A useful set of controlled comparisons is:

1. **Chronological sequence:** provide the sampled frames in their original order.
2. **Reversed sequence:** provide exactly the same frames in reverse order.
3. **Shuffled sequence:** provide exactly the same frames in a random order, repeating the shuffle with multiple fixed seeds.
4. **Order-agnostic control:** mean-pool the same frame embeddings, removing explicit order information.
5. **Matched-frame single-frame controls:** compare against individual frames selected from the same observation budget, so the temporal model is not credited merely for receiving more images.

If performance falls reliably when the same frames are reversed or shuffled, while the chronological sequence performs better, that is evidence that order carries useful information. If chronological, reversed and shuffled sequences perform similarly, the model may primarily be aggregating visual content rather than exploiting temporal relationships.

The project includes chronological, reversed and 20-permutation shuffled evaluations, as well as mean pooling as an order-agnostic control. However, the observed order-related differences are small, so the current results provide only limited evidence of strong temporal-order dependence.

## D. What experimental result would convince you that the temporal approach is not useful for the selected task?

I would consider temporal modeling unsupported for this task if, under repeated and controlled video-level splits:

- Temporal models consistently fail to outperform a well-tuned single-frame baseline by a meaningful margin.
- Their gains disappear when the single-frame baseline is given a comparably informative frame or when frame count is controlled.
- Chronological, reversed and shuffled versions of the same frames produce statistically indistinguishable results.
- Mean pooling performs as well as order-aware models, indicating that explicit temporal modeling adds no measurable value.
- Any apparent improvement is inconsistent across random seeds, classes or held-out splits.

The SigLIP results already provide an example in this direction: the single-frame model outperformed all tested temporal architectures. The CLIP results, however, prevent a project-wide conclusion that temporal modeling is useless. The more defensible conclusion is that its value depends on the visual encoder and must be demonstrated against controlled baselines.

## E. What would you change in the dataset or experiment to obtain stronger evidence?

I would make the following changes:

1. **Expand the dataset.** The current experiment uses a partial nine-class HMDB51 subset containing 950 videos. I would use more of HMDB51, ideally the complete set of classes and more videos per class, to test whether the findings generalize beyond this subset.

2. **Improve class coverage.** The current subset is imbalanced: for example, `clap` has 130 videos while `draw_sword` has 48. I would increase the number of examples in underrepresented classes and report per-class metrics alongside macro-F1.

3. **Prioritize actions that require motion.** I would include or emphasize action pairs that are visually similar in individual frames but differ in movement or event order. Such examples make it easier to test whether temporal relationships provide information that a static frame cannot provide.

4. **Use multiple predefined video-level splits and random seeds.** The current split is video-level (664 training, 143 validation and 143 test videos), which prevents frames from the same video crossing splits. Repeating this evaluation across several stratified video-level splits would show whether the observed differences are stable rather than split-specific.

5. **Use strictly matched frame inputs for order tests.** For each video, select one fixed set of frames and evaluate chronological, reversed and shuffled orders using that exact set. Repeat shuffling with multiple seeds and report the distribution of results. This isolates order from frame content.

6. **Add stronger frame-budget controls.** Compare temporal models with single-frame models evaluated on each of the same sampled positions, as well as with order-agnostic pooling. This helps separate the benefit of additional observations from the benefit of temporal reasoning.

7. **Test more observation budgets consistently.** Evaluate models using increasing numbers of frames and report the performance-versus-frame-count curve. This would clarify whether improvements come from seeing more of the video or from modeling relationships between frames.

8. **Report uncertainty and class-level behavior.** Repeat training where feasible, report variation across seeds or splits, and inspect per-class confusion matrices. Aggregate accuracy alone can hide classes for which motion helps and classes for which it does not.

## Overall conclusion

The experiments suggest that temporal information can improve video classification, but the benefit is **not universal**. The strongest result came from the CLIP Transformer, while the SigLIP single-frame model performed better than every tested SigLIP temporal model. Furthermore, the small chronological-versus-shuffled differences indicate that the current experiments do not provide strong evidence that correct temporal order is the main source of the CLIP Transformer's improvement.

The most accurate conclusion is therefore that **multi-frame information helped in some settings, but the specific contribution of temporal relationships remains only partially established**. A larger and more motion-discriminative dataset, repeated video-level splits, and strict same-frame order controls would provide stronger evidence.
