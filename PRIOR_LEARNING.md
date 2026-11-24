# VSA closure with Hebbian, competitive and Hopfield learning

These experiments extend the [counted-prior baseline](BASELINE.md) using the shared face renderer.
Start with **computationalExperiments/prior_learning/supervised_hebbian_category_learning.ipynb** for the new online training experiment.
Open **computationalExperiments/prior_learning/competitive_hebbian_category_learning.ipynb** for label-free competitive learning,
multiple random seeds, unit-usage diagnostics and held-out completion comparisons.
Open **computationalExperiments/prior_learning/hopfield_exemplar_memory.ipynb** for the energy-based exemplar prior, the kernel
reference, and recurrent completion of partial feature cues.
The experiments use the current version 2 renderer with
729 face recipes. Eye shape has three values: dots, arches, and oval pupils.
The outline is closed across the current vocabulary.

The new `SupervisedHebbianPrior` strengthens category–feature connections from
one-hot co-activation. Training receives category labels and feature values.
Normalized connection strengths produce the category prior used in completion.
At learning rate 1 with matching pseudocounts, one pass reproduces batch counts.

The full learned-prior notebook defaults to `CATEGORY_LEARNING = "hebbian"`.
Set it to `"counts"` for the batch reference or `"competitive"` for anonymous
units learned without category labels. The competitive backend exposes its unit
count, epochs, learning rate, temperature and homeostatic inhibition strength.
In this mode, unit cues in the full notebook refer to anonymous units.
The supervised learning-rate
setting changes the relative strength of observations and initial pseudocounts.
The other prior families retain their existing fitting methods.

```python
from contour_models.hebbian_category_priors import SupervisedHebbianPrior
from contour_models.face_recipes_and_observations import CARDINALITIES, CLASS_NAMES

learner = SupervisedHebbianPrior(CARDINALITIES, len(CLASS_NAMES))
learner.update([1, 2, 2, 2, 0, 0], category=0)
print(learner.feature_probabilities[3])  # category-specific nose probabilities
```

To run the notebooks and tests from the project root, use

```powershell
python tools/notebooks/execute_notebook.py computationalExperiments/prior_learning/supervised_hebbian_category_learning.ipynb
python tools/notebooks/execute_notebook.py computationalExperiments/prior_learning/competitive_hebbian_category_learning.ipynb
python tools/notebooks/execute_notebook.py computationalExperiments/prior_learning/hopfield_exemplar_memory.ipynb
python tools/notebooks/execute_notebook.py computationalExperiments/face_completion/face_completion_learned_priors.ipynb
python -m unittest -v tests.test_face_completion tests.test_supervised_hebbian_prior tests.test_competitive_hebbian_prior tests.test_hopfield_exemplar_prior
```

The dedicated notebook saves its figures, a comparison summary, and learned
weights under `outputs/hebbian/`. Rebuilding it with
`python tools/notebooks/build_supervised_hebbian_category_learning.py` replaces that notebook with an unexecuted copy.
Save notebook edits before rebuilding.

## Competitive learning

`CompetitiveHebbianPrior` uses only feature rows during learning. Positive random
weights break unit symmetry. Units compete through their feature likelihoods and
a penalty for recent overuse. Co-activation drives an activity-dependent normalized
Hebbian update. Feature observations and the number of units remain supplied.
The update does not guarantee category discovery or monotonically improving loss.

```python
from contour_models.hebbian_category_priors import CompetitiveHebbianPrior
from contour_models.face_recipes_and_observations import CARDINALITIES

learner = CompetitiveHebbianPrior(CARDINALITIES, n_units=3, seed=0)
learner.fit(training_features, epochs=20, shuffle_seed=0)
logprior = learner.log_prior(candidate_features)
```

The notebook uses synthetic category labels after training to evaluate agreement
under a one-to-one unit/name alignment. It reports every fixed random seed and uses
the first for its example. It compares completion against uniform and supervised
priors on identical fragments. A homeostasis ablation and hard/soft usage reports
help reveal duplicate or inactive units. Reports and weights go into
`outputs/competitive/`. Run `python tools/notebooks/build_competitive_hebbian_category_learning.py` to rebuild the
unexecuted notebook, saving any notebook edits first.

## Hopfield exemplar memory

`HopfieldExemplarPrior` learns from the same unlabeled training configurations
as the kernel exemplar reference. It encodes six categorical feature groups as
18 bipolar units. Hebbian weights average the outer products and have zero
self-connections. Candidate probabilities normalize `exp(-beta * energy)` over
the same valid drawing vocabulary. The mean-weight normalization makes duplicating
the entire training corpus leave the energy unchanged. Repeated individual
examples still affect their relative contribution.

```python
from contour_models.hopfield_exemplar_prior import HopfieldExemplarPrior
from contour_models.face_recipes_and_observations import CARDINALITIES

memory = HopfieldExemplarPrior(CARDINALITIES, inverse_temperature=1.0)
memory.fit(training_features)
logprior = memory.log_prior(candidate_features)
result = memory.retrieve([1, 2, -1, -1, 0, -1], seed=0)
print(result["state"], result["energy_trace"])
```

The full learned-prior notebook keeps `EXEMPLAR_LEARNING = "kernel"` by default.
Choose `"hopfield"` and adjust `HOPFIELD_INVERSE_TEMPERATURE` to use the energy
prior. The fitted `bank.exemplar_model` also exposes recurrent retrieval.

Retrieval accepts partial **feature** cues, with `-1` for unknown features or
an explicit known-feature mask. It updates whole groups to keep exactly one
value per feature, clamps known values and keeps the current value on ties.
Strictly improving asynchronous updates decrease energy, but can reach an
incorrect stored configuration or a novel combination. The notebook compares
these endpoints with exhaustive minimum-energy and kernel-MAP references.
Image completion separately combines the learned prior with the existing
pixel/VSA evidence. Category labels enter neither memory training nor retrieval.

Reports, figures, and learned weights go into `outputs/hopfield/`.
Run `python tools/notebooks/build_hopfield_exemplar_memory.py` to rebuild its unexecuted notebook,
saving any notebook edits first. Prior entropy and inverse temperature should
be considered when comparing the two exemplar models.

## Full experiment with learned priors

Open **computationalExperiments/face_completion/face_completion_learned_priors.ipynb** and run all cells. It ships with executed outputs.
The first parameter cell controls vector dimensions, prior strength, noise,
visibility, and number of trials. No network, API key, or GPU is needed.

The notebook tests completion and invented contours in synthetic binary cartoons.
It learns priors from separate training configurations and keeps the test category
hidden. Four prior families share the same candidate vocabulary. VSA-compressed
image evidence is compared with an exact pixel baseline; dense and VSA-encoded
feature messages are compared during iterative inference.

This is a controlled closure analogue with a known missing-pixel mask, **not** a
model validated on the cow photograph or on real Mooney images. The shape grammar
is supplied. Category training depends on the selected backend. See the notebook for limitations,
equations, controls, metrics, references, and suggestions for extending the model.

## Six-hint comparison against the poster simulation

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
python computationalExperiments/face_completion/compare_priors_on_six_hint_trials.py
```

The script compares batch counting, supervised Hebbian category fitting,
label-free competitive Hebbian fitting, and Hamming-kernel and Hopfield
exemplar fitting. The competitive learner sees the same 750 feature rows
with the category labels withheld. It uses three anonymous units and 20
passes, repeated with initialization and training-order seeds 0, 1, and 2.
All three runs reuse the same test trials. The script compares the two
exemplar priors after matching their effective entropy at the requested prior
weight. That calibration only uses the fitted priors. It uses 12 held-out
faces per condition, the same masks and 5% pixel flips across methods, six
hints with 0–6 correct, and a default prior weight of 12. The accuracy tally
is hidden from the model. A score of zero correct hints means six wrong hints.

Figures, per-trial scores, condition means, weight sensitivity, and the run
configuration go into `outputs/prior_comparison/`. The competitive contour
shows the seed mean and all three individual runs. To check the counting
predictions against the saved poster grid, pass
`--reference outputs/poster/plot_data/counting_reference_trials.json`.

### Visibility and hints with pixel flips capped at 20%

Run `python computationalExperiments/face_completion/compare_priors_under_pixel_flips.py` to compare feature counts and label-free
Hebbian learning across 2%, 5%, 10%, 15%, and 20% pixel flips. Each condition
uses the same 64 held-out faces by default. The simulator flips the requested
share of visible pixels, rounded down, so each image stays at or below 20%.
Both methods see the same masks, flips, and hints at prior weight 12.

`outputs/capped_noise_contours/` contains the contour averaged equally across
the five rates, the contours for each rate, and their source means.

## Files

- `computationalExperiments/face_completion/face_completion_learned_priors.ipynb`: experiment and tuning controls, with plots.
- `contour_models/face_recipes_and_observations.py`: renderer, train/test split, learned priors, metrics.
- `contour_models/vsa_inference.py`: role binding, image encoding, probabilistic/variational inference.
- `tests/test_face_completion.py`: numerical and leakage checks.
- `tools/notebooks/execute_notebook.py`: optional in-process IPython runner, capturing cell outputs without kernel sockets.
- `outputs/learned_priors/`: paired benchmark CSV, figures, and run summary generated by the notebook.

## Environment and checks

Use an existing Python/Jupyter environment with the packages in `requirements.txt`.
If needed, run `python -m pip install -r requirements.txt` in an environment you manage.
For the notebook UI, use JupyterLab/Notebook or an editor with Jupyter support.

From this directory:

```powershell
python -m unittest -v tests.test_face_completion
```

The benchmark is exploratory. Selecting a winning setting from its results requires
fresh validation trials before making a generalization claim. A lower posterior
entropy is not, by itself, evidence of better completion.
