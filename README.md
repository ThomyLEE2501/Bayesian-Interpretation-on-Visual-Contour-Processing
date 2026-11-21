# BayesContourProcessing

Can knowing that a patchy image shows a cow change which marks you group into its muzzle? This project studies a tractable part of that question. People and a Bayesian model reconstruct the six features of a masked, noisy drawing. Visibility, pixel flips, learned expectations and correct or misleading hints can all be varied. The [daily project log](PROGRESS.md) traces the work from the cow photograph through the pilot results and includes the chat diagrams.

| Start here | Contents |
| --- | --- |
| [Poster](Poster/Bayesian_Visual_Contour_Poster_Hebbian_Rule.pptx) | Final editable November 20 poster with model and human results. Its [plot data](outputs/poster/plot_data/) include the filtered human series and the high-visibility model extension. |
| [Writeup](writeup/closure_experiment.pdf) · [LaTeX](writeup/closure_experiment.tex) | Current 729-recipe design, equations, prior-learning methods, results and encoder proposal. The source figures are beside the TeX file in [figures](writeup/figures/). |
| [Baseline notebook](vsa_gestalt_closure.ipynb) · [baseline guide](BASELINE.md) | Counted priors, pixel and vector evidence, full-candidate and iterative inference. [Saved runs](outputs/) include the November 16 visibility, targeted-cue and mixed-cue experiments. |
| [Prior-learning guide](PRIOR_LEARNING.md) | Executed notebooks for [supervised Hebbian](hebbian_learning.ipynb), [competitive Hebbian](competitive_learning.ipynb) and [Hopfield](hopfield_learning.ipynb) learning. [Results](outputs/) include weights, matched comparisons and capped-noise sweeps. |
| [Human study guide](psychophysicalExperiment/README.md) · [Windows launcher](<psychophysicalExperiment/Start face study.cmd>) | Desktop and [browser experiment](psychophysicalExperiment/web/index.html), response files, feedback images, session schedules and analysis tools. |
| [Presentation](presentation/VSA_Contour_Completion_revised.pptx) | Latest teaching deck. It describes the November 9 prototype with 486 recipes. The poster and writeup use the later 729-recipe vocabulary. |

## Working from this directory

Open the notebooks from this directory. The baseline notebook selects batch counting. The [full learned-prior notebook](vsa_gestalt_closure_learned_priors.ipynb) defaults to supervised Hebbian learning and exposes the competitive and Hopfield settings. Both use the same renderer and inference code. With the [dependencies](requirements.txt) installed, run the test suite from here.

~~~powershell
python -m unittest discover
~~~

All saved runs live under [outputs](outputs/). The [baseline](outputs/baseline/) and [learned-prior](outputs/learned_priors/) benchmark folders each contain their own metrics, figures and run summary. Named experiment folders hold the sweeps and prior comparisons. [Psychophysics](outputs/psychophysics/) holds the participant sessions and accuracy plots. [Poster data](outputs/poster/) holds the recovery sets and plotted series.

## Which results were kept

The saved [baseline benchmark](outputs/baseline/benchmark.csv) used renderer version 1. The current baseline notebook uses version 2, which adjusts the dot-eye positions. To reproduce the archived CSV, pass `renderer_version=1` to `make_library` in the baseline notebook's library setup cell. The [learned-prior benchmark](outputs/learned_priors/benchmark.csv) uses version 2.

For each experiment, the collection keeps the latest saved run, its inputs or configuration, figures, tables and executed notebooks. The November 16 eye-shape runs replace the November 13 versions. The capped-noise and high-visibility runs cover different conditions, so both are present. The [poster](Poster/) uses set 06 of the seven [sampled recovery examples](outputs/poster/Face_Recovery_Sets/).

The two complete pilot sessions each have 50 responses. The two partial sessions have five and two confirmed responses. The raw session folders stay local under `outputs/psychophysics/participant_sessions/`. The poster pools the complete sessions and applies its two-trial exclusion only to the analysis. The [pooled plot data](outputs/poster/plot_data/human_scatter_data.json) contain the 98 trials used in the poster.

The source folders were `C:\NEU_172L\vsa_gestalt_closure` and `C:\NEU_172L\vsa_gestalt_closure_hebbian`. The revised teaching deck came from their sibling `outputs` folder. Chat figures are embedded or linked in the log.
