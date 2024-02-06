---
title: Applications of data valuation
---

# Applications of data valuation

Data valuation methods hold promise for improving various aspects
of data engineering and machine learning workflows. When applied judiciously,
these methods can enhance data quality, model performance, and cost-effectiveness.

However, the results can be inconsistent. Values have a strong dependency
on the training procedure and the performance metric used. For instance,
accuracy is a poor metric for imbalanced sets and this has a stark effect
on data values. Some models exhibit great variance in some regimes
and this again has a detrimental effect on values.

While still an evolving field with methods requiring careful use, data valuation can
be applied across a wide range of data engineering tasks. For a comprehensive
overview, along with concrete examples, please refer to the [Transferlab blog
post]({{ transferlab.website }}blog/data-valuation-applications/) on this topic.

## Data Engineering

While still an emerging field, judicious use of data valuation techniques
has the potential to enhance data quality, model performance,
and the cost-effectiveness of data workflows in many applications. 
Some of the promising applications in data engineering include:

- Removing low-value data points can reduce noise and increase model performance.
  However, care is needed to avoid overfitting when iteratively retraining on pruned datasets.
- Pruning redundant samples enables more efficient training of large models.
  Value-based metrics can determine which data to discard for optimal efficiency gains.
- Computing value scores for unlabeled data points supports efficient active learning.
  High-value points can be prioritized for labeling to maximize gains in model performance.
- Analyzing high- and low-value data provides insights to guide targeted data collection
  and improve upstream data processes. Low-value points may reveal data issues to address.
- Data value metrics can also help identify irrelevant or duplicated data
  when evaluating offerings from data providers.

## Model development

Data valuation techniques can provide insights for model debugging and interpretation.
Some of the useful applications include:

- Interpretation and debugging: Analyzing the most or least valuable samples
  for a class can reveal cases where the model relies on confounding features
  instead of true signal. Investigating influential points for misclassified examples
  highlights limitations to address.
- Sensitivity/robustness analysis: Prior work shows removing a small fraction
  of highly influential data can completely flip model conclusions.
  This reveals potential issues with the modeling approach, data collection process,
  or intrinsic difficulty of the problem that require further inspection.
  Robust models require many points removed before conclusions meaningfully shift.
  High sensitivity means conclusions heavily depend on small subsets of data,
  indicating deeper problems to resolve.
- Monitoring changes in data value during training provides insights into
  model convergence and overfitting.
- Continual learning: in order to avoid forgetting when training on new data,
  a subset of previously seen data is presented again. Data valuation helps
  in the selection of highly influential samples.

## Attacks

Data valuation techniques have applications in detecting data manipulation and contamination:

- Watermark removal: Points with low value on a correct validation set may be
  part of a watermarking mechanism. Removing them can strip a model of its fingerprints.
- Poisoning attacks: Influential points can be shifted to induce large changes
  in model estimators. However, the feasibility of such attacks is limited,
  and their value for adversarial training is unclear.

Overall, while data valuation techniques show promise for identifying anomalous
or manipulated data, more research is needed to develop robust methods suited
for security applications.

## Data markets

Additionally, one of the motivating applications for the whole field is that of
data markets: a marketplace where data owners can sell their data to interested
parties. In this setting, data valuation can be key component to determine the
price of data. Market pricing depends on the value addition for buyers
(e.g. improved model performance) and costs/privacy concerns for sellers.

Game-theoretic valuation methods like Shapley values can help assign fair prices,
but have limitations around handling duplicates or adversarial data.
Model-free methods like LAVA [@just_lava_2023] and CRAIG are
particularly well suited for this, as they use the Wasserstein distance between
a vendor's data and the buyer's to determine the value of the former. 

However, this is a complex problem which can face practical banal problems like
the fact that data owners may not wish to disclose their data for valuation.