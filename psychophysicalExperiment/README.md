# Face reconstruction study

## Browser experiment

[Open the experiment on GitHub Pages](https://thomylee2501.github.io/Baysian-Intrepretation-on-Visual-Contour-Processing/).

Open the link in desktop Chrome or Edge, enter a participant ID and session number, then select **Choose local folder** and grant write access. The study creates a fresh session subfolder. Each confirmation saves its feedback PNG and updates `responses.csv`. The introductory slides, a reproducible `session.json` and the final accuracy figure are also saved locally. An existing session folder is protected against overwriting.

The **Download a session ZIP instead** option works in browsers without writable directory access. Download from any feedback screen to keep a partial session or from the final screen to keep the complete session. In ZIP mode the tab holds the data in memory until you download it. Participant responses and file contents stay on the device.

The browser uses the same 729 face images and the same held-out recipe set as the desktop study. It uses its own seeded random generator and records the actual mask, observed pixels and hint values for every trial. The desktop replay command below reads these directly. A browser seed reproduces another browser session with the same settings.

The static site lives in `web/`. The published copy resides at the root of the repository’s `gh-pages` branch. After changing the Python renderer or feature guide, run `python -m psychophysicalExperiment.export_web_assets` from the repository root to rebuild `web/assets.js`. Participants only need the browser link.

## Desktop experiment

On Windows, open **Start face study.cmd** to run the participant slideshow. Open **Plot face accuracy.cmd** to choose CSV files, axes, filters and an output figure. Run the terminal examples in this guide from the `psychophysicalExperiment` folder. Install the shared dependencies with `python -m pip install -r ../requirements.txt`, then run `python face_slideshow.py` or `python choose_contour.py`.

This folder contains the slideshow and analysis programs. Sessions and comparison plots are stored under [psychophysics outputs](../outputs/psychophysics/). The face renderer and VSA model are shared with the computational notebook in the parent folder. Open the Windows launcher in this folder to start the local study.

## Running a session

1. Enter the participant's name or study ID and a positive session number. Open **Configure session hints** to change the range, reliability or tally disclosure before starting. The program creates `../outputs/psychophysics/participant_sessions/<name>_session_<number>/`. Choose a fresh number for each session.
2. Advance through the 18-option feature guide and the three typical examples. The guide and response panel show head shape, ear shape, eye shape, eye spacing, nose shape and layout in that order. All 729 recipes are possible within the face grammar.
3. Inspect the masked image on the left. The percentage below it is the fraction of all image pixels revealed on this trial, including visible pixels affected by noise. Underlined choices mark the hints, which can be incorrect. When the tally is enabled, **Hint accuracy** gives the aggregate counts. Choose one option per row and press **Confirm recipe**.
4. The final slide shows session accuracy. Confirmed responses are saved after every trial, including in a session that ends early. The program protects an existing session folder against overwriting.

The slideshow starts maximized on Windows. The evidence, feature examples, slide images and text resize with the window. Each session draws 50 distinct recipes uniformly from the model's 182 held-out recipes. New sessions use these counts of **actually visible** pixels:

| Visible pixels | Default trials |
|---|---:|
| Under 20% | 5 |
| 20% to under 35% | 10 |
| 35% to under 60% | 20 |
| 60% and above | 15 |

Set the four counts in **Experiment settings** in the browser or **Configure visibility and noise** on desktop. Counts must sum to 50 and the band boundaries are fixed. The study samples exact counts of visible 5×5 blocks within each band, then randomly chooses their locations. Each trial shows at least one block. The maximum pixel-flip setting limits the available noise levels from 2%, 5%, 10%, 15% and 20%. Noise levels and hint totals are each balanced within a band as evenly as its trial count allows. Trial order is shuffled.

For a total of $n$ hints, the number correct is drawn uniformly from 0 through $n$ by default. The generator randomly selects which features receive hints and which hints are correct. An incorrect hint names one of the feature’s two alternatives with equal probability. Hints always appear in feature-guide order.

| Session option | Effect |
|---|---|
| `--visibility-counts 5 10 20 15` | Set trial counts in the four fixed visibility bands. |
| `--noise-cap 10` | Sample only flip probabilities of 2%, 5% and 10%. The default cap is 20%. |
| `--hint-range 2 5` | Vary the number supplied from two through five. Use `--hint-range 4 4` for a fixed total. |
| `--hint-accuracy 0.7` | Draw the correct count from a binomial distribution with 70% reliability per hint. Values of 0 or 1 produce all-wrong or all-correct hints. |
| `--hide-hint-counts` | Present the same seeded hints with the tally hidden. |
| `--uncued` | Run an image-only session. |
| `--seed 172` | Reproduce a schedule. Reuse the seed across hint conditions to keep the same faces, masks, noise and trial order. By default the program generates and saves a seed. |
| `--sessions PATH` | Choose the output location. |

## Session files

- `feature_guide.png` and `typical_faces.png` are the two introductory slides. The layout row uses a faint head outline as a position reference.
- `trial_01.png` through `trial_50.png` are the feedback slides.
- `responses.csv` has one row per confirmed trial. It records the subject, session, seeds, recipe, requested and realized visibility and noise, true and chosen features, six correctness flags, total score, response time and feedback filename. The six `cue_*` fields contain the supplied values with an empty field for each omitted feature. The columns `cues_given`, `correct_cues`, `incorrect_cues` and `hint_counts_shown` record the experimental condition. The timer runs from trial display to confirmation.
- `session.json` contains the session seed, renderer version and full schedule. Version 2 uses inset dots at three distinct spacings. Replay selects the original geometry for earlier sessions without a version field. Numeric recipe IDs and CSV feature columns retain their original order. `session_accuracy.png` displays the results of a completed session. Participant names, responses and feedback are local to the session folder.
- Run `python face_accuracy.py replay "../outputs/psychophysics/participant_sessions/NAME_session_001" --alpha 12` for MAP inference on confirmed trials. The resulting `model_replay.csv` uses the participant’s image, hints and disclosure condition. With a disclosed tally of $k$, a candidate must match exactly $k$ of the supplied hints. All candidates satisfying the count receive equal hint likelihood because the subset of correct hints was drawn uniformly. With the tally hidden, `--beta` scales the model’s assumed 95%-reliable hint likelihood. Older saved sessions replay with their original six-hint or image-only schedule and hidden tallies.

## Plotting

The plot picker accepts one or multiple response files. Choose X and Y, optionally enter filters such as `cues_given=4,hint_counts_shown=1`, and save. Select **Logarithmic visibility axis** to expand the low-visibility range on either axis. Tick labels remain percentages. Zero visible pixels can occur under block masking. The log plot reports how many such observations were omitted, and the linear view includes them.

Each dot represents one six-feature response. Scores of **0–4** use dark to light orange, **5** uses cyan and **6** uses deep teal. Choosing uniformly among three options per feature gives an expected score of **2/6**. The color boundary at five marks high recovery. Contours interpolate the mean score between measured coordinates in the selected axis scale. Dots at repeated coordinates are offset slightly. To pool participants, select their CSVs together. For a matched human/model comparison, plot each source with the same filters and axes. Holding `cues_given` fixed distinguishes the effect of correct hints from simply receiving more hints.

The examples under `../outputs/psychophysics/accuracy_contours/` include visibility against correct hints (linear and log), noise and prior weight. The earlier three-hint experiment is also rescored on all six features. `prior_effect.png` and `prior_effect.csv` report the fraction of trials where the chosen recipe changed from the matched zero-prior decision. The CSV also gives the mean recovery score at each visibility and weight.

```powershell
python face_accuracy.py simulate
python face_accuracy.py plot ../outputs/psychophysics/accuracy_contours/model_scores.csv --x visibility_pct --y correct_cues --where prior_weight=12 --where noise_pct=5 --log-visibility --out cues.png
python face_accuracy.py plot responses.csv --x actual_visibility_pct --y flipped_pct --out actual.png
python face_accuracy.py import-mixed ../outputs/mixed_feature_cues_2026-11-16
```

The default sweep crosses 12 held-out recipes, five visibilities, five noise levels, seven correct-hint counts and prior weights of 0, 1, 3, 8, 12, 16 and 24. All six features are hinted so that moving from $k$ to $k+1$ correct replaces one wrong hint. Within a recipe, the same random draws control block selection, noise and alternative hints. The worked plots use a fixed alpha of 12 across visibilities. On seed 172 this changes the MAP in at least one-third of matched cases at every tested visibility. This setting illustrates prior influence. To examine accuracy, use the six-feature score or the `mean_score` column. Use `--prior-weights`, `--beta`, `--seed` and `--repeats` to vary the sweep.

The model combines VSA image evidence, the fitted category-mixture prior raised to alpha, and the hint likelihood. The default sweep hides the tally and assumes 38-to-1 odds for a named value versus one alternative. Run `simulate --show-hint-counts` to give the model the tally, or use `replay` for the exact variable-hint human schedule. Alpha scales the learned face prior. Beta applies only to hidden-tally hints and its CSV field is empty when a tally was disclosed.
