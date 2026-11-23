# Daily project log

This log uses Pacific dates and has one section per day with a recorded project update. “You” are Thomy and “I” am Codex. The entries reconstruct the discussion from **Describe what you see**, its three project side chats, **Explain skipped MLE steps**, and **Inference Analogy Comparison**. Diagrams retain their original vocabulary. Links to experiment files open the most recent retained version.

## November 3, 2026 — Seeing the cow

**You asked** me to describe the image without searching the web, explain why a face came to mind, and consider whether feedforward evidence and feedback could account for recognition. You asked for a notebook where you could change how strongly a prior influenced the contours drawn into a gap.

**I did** identify a cow facing the viewer. The pale central strip, flanking dark patches and broad lower muzzle supported that reading. We worked from that observation toward a synthetic experiment in which the true missing strokes would be known. I started building the notebook that evening.

**What needed an ELI5 explanation.** What does a “role” do, and how could a dot product mean uncertainty? Think of a role as the label on a slot, such as *nose shape*. Binding keeps a feature attached to that slot. For a bipolar key $r$, multiplying by $r$ twice undoes the binding because each entry squares to one. Several proposed shapes can agree with the same patch. Turning their similarities into probabilities requires a model of how variable or noisy the evidence is.

**Your new ideas and questions.** You proposed recovering several factors within a single bound part, such as shape, color and position relative to the eyes. Your motivating observation was behavioral: once people recognize the cow, their sketches can gain contours that the photograph never explicitly supplied. Would a model invent those contours too?

**My clarifications.** The cow description gives visual reasons for an interpretation. It cannot establish the internal sequence by which recognition happened. A model should allow uncertainty to remain high when multiple explanations fit. A confident guess alone would be a poor test of closure.

![Ambiguous cow image that started the project](assets/ambiguous-cow.png)

Your reference diagram for the question about feedback between levels:

![Original hierarchical feedback sketch supplied in the main chat](diagrams/hierarchical-feedback-sketch.png)

## November 4, 2026 — First runnable experiment

**You asked** me to push the notebook and supporting files to the GitHub repository you specified, initially private.

**I did** finish and execute the first notebook after midnight. It varied visibility, pixel noise, prior strength and vector dimension. It compared exact-pixel and VSA evidence, included a misleading-cue condition and tested genuinely open contours. After you completed sign-in, I pushed it to the private repository. [Current baseline notebook](vsa_gestalt_closure.ipynb)

**What needed an ELI5 explanation.** No new conceptual question was recorded during the repository handoff. The first experiment gave the sketch question two separate scores. Filling a hidden stroke earns credit if a stroke really belongs there. Drawing across a truly empty gap counts as an invention. One reconstruction can do both.

**Your new ideas and questions.** You chose private access for this stage of the project.

**My clarifications.** The model received a mask showing where evidence was missing, a vocabulary of possible parts and a renderer that could draw them. Results on those synthetic images answered a narrower question than recognizing the original photograph.

The main chat illustrated this comparison with a completion panel. This is the retained version using the current face vocabulary.

![Completion under different priors, from the current baseline run](outputs/baseline/completion_demo.png)

## November 5, 2026 — What the model learns before seeing a fragment

**You asked** how training worked, whether VSA did the training, and for a plain-language walk through the layout, category and exemplar priors in `fit_priors()`. In a related chat you connected the face experiment to maximum a posteriori estimation (MAP).

**I did** trace the code from training feature rows to one prior score per possible drawing. We separated the drawings used for fitting from the combinations held out for reconstruction. [Baseline implementation](closure_model.py)

**What needed an ELI5 explanation.** Imagine a pile of recipe cards. A layout prior counts how often arrangements occur. A category prior counts how often each ingredient occurs within a group such as cow-like faces, then can mix across groups. An exemplar prior asks how closely a recipe resembles whole training examples. A uniform prior gives every recipe the same starting weight. Adding a small count to every option keeps an unseen option possible.

**Your new ideas and questions.** You recognized that the experiment asks which face is plausible *after* seeing its fragment. Could fitting parameters by maximum likelihood and choosing a drawing by MAP both belong in the same experiment?

**My clarifications.** Yes. Model fitting and choosing a test drawing answer different questions. MAP chooses the recipe maximizing likelihood times prior. The full posterior also tells us how much probability its competitors retain. In the original fitter, training used supplied feature values. VSA entered when scoring image evidence or passing feature messages.

## November 6, 2026 — From a recipe table to a top-down message

**You asked** what a candidate was, how the scoring rule was learned, what `samples[:, 5]`, `states[:, j]` and `normalize_logp` did, and whether the bank of priors changed at test time. You asked especially how the other features could influence a guess about the nose.

**I did** follow one candidate through counting, indexing, normalization and inference, and drew the two routes from shared evidence to a predicted stroke.

**What needed an ELI5 explanation.** Each row is a face recipe. Column $j$ lists one feature for every face. Counting a training column makes a lookup table. Reading that table with the candidate column gives each face the score for its own value. Given log scores $s_i$, subtracting $\log\sum_k e^{s_k}$ makes the exponentiated scores sum to one. During a trial the model keeps the learned table fixed and updates its beliefs about this face. A nose gets more support when it goes well with the ears and head that currently look plausible.

**Your new ideas and questions.** You connected the image score to a likelihood and saw why adding log prior and log likelihood gives a posterior score. You then asked where VSA contributes to each of those computations.

**My clarifications.** The renderer draws pixels from a recipe. Inference works back toward plausible recipes by comparing drawings with the visible marks. Full-candidate inference normalizes scores over complete faces. Iterative inference updates feature distributions using the current beliefs about the other features.

```mermaid
flowchart LR
    A[Training drawings] --> B[Prior score per candidate]
    C[Visible pixels and mask] --> D[Compare with rendered candidates using PixelVSA]
    D --> E[Evidence score per candidate]
    B --> F[Normalize over complete candidates]
    E --> F
    B --> G[Iteratively update feature probabilities]
    E --> G
    F --> H[Predict hidden strokes]
    G --> H
```

## November 9, 2026 — What changes when someone says “cow”?

**You asked** for a 15-slide presentation, the axes and missing dense curve on the convergence plot, and an explanation of how full-candidate and iterative inference relate. In the side chats you asked about masks, cross-talk, softmax, feature dependence, unbinding, sparse memory and the breadth-first/depth-first analogy.

**I did** build the deck and revise slide 12 so the nearly overlapping curves could be read. The horizontal axis counts sweeps through all six features. The vertical axis shows variational free energy in nats. The final dense and VSA values were about 81.804 and 81.900. I also drew a proposed recurrent encoder. [Revised presentation](presentation/VSA_Contour_Completion_revised.pptx)

**What needed an ELI5 explanation.** A white pixel you can see rules out a black stroke there. A hidden pixel tells you nothing either way, so it needs a separate mask. Full-candidate inference gives every complete recipe a score and uses softmax to turn those scores into probabilities. Iterative inference keeps several possibilities alive per feature and revisits them. If seeing a muzzle increases belief in a cow-like category, ears common in that category can gain support too.

**Your new ideas and questions.** What happens if the visible muzzle has a shape the renderer cannot draw? Would adding more and more predefined recipes just move the scalability problem? You identified the original target more precisely: a “cow face” cue might change how the *same marks* are grouped. You asked whether a learned encoder or associative memory could recover that structure directly.

**My clarifications.** Both inference methods use the same learned prior and image scores and produce separate answers. The full-candidate result does not prune the iterative search. This implementation still consults the recipe table for its iterative updates. “Exhaustive scoring” and “repeated feature updates” describe the computations more precisely than tree-search terminology. A cue-conditioned encoder was a proposal for a different experiment.

```mermaid
flowchart LR
    A[Image and mask] --> B[Visual encoder]
    B --> C[Recurrent belief state]
    D[Prior cue] --> C
    C --> E[Feature probabilities]
    E --> F[Image decoder]
    F --> G[Compare prediction with visible image]
    G --> C
```

## November 10, 2026 — Making cues visible and learning priors online

**You asked** me to fix notebook math delimiters, preserve block masks, keep the notebook builder locally, and investigate why the example face looked perfect under every cue. You asked me to try different seeds, visibilities and prior weights. In parallel, you requested a pipeline diagram and a local copy for supervised and label-free Hebbian learning.

**I did** verify that the posterior was recomputed for each cue. I increased the precision of its display and found an illustrative mask with seed 67. A target visibility of 47% revealed 62.5% of this particular image. With cue strength 3, correct and wrong cues produced different eye spacings despite equally good visible fits. I kept the 0–24 sweep. In the learning copy, one-pass supervised Hebbian learning reproduced the batch-count result at matching settings. The competitive notebook compared three training seeds and recorded how often each unit was used. [Supervised notebook](hebbian_learning.ipynb) · [Competitive notebook](competitive_learning.ipynb)

**What needed an ELI5 explanation.** A printed probability of 1.00 can hide a value such as 0.99999954. A mask can reveal enough distinctive marks to identify one recipe among 486. For supervised Hebbian learning, a category and its observed features are active together, strengthening their connections. With competition, several anonymous units respond and the more active ones learn more from that face.

**Your new ideas and questions.** Could the demonstration keep the fragment recognizable to a human while leaving one feature genuinely ambiguous? Could category preferences emerge without supplying the category names?

**My clarifications.** The chosen seed illustrates a mechanism. The sweeps test how often it appears. Removing category labels still leaves supplied feature values and a chosen number of units. In your pipeline sketch, the original vocabulary had 486 candidates, and both the observation and every candidate must receive the same mask before encoding.

Your full-candidate sketch and the companion iterative diagram from the side chat:

![Original full-candidate inference sketch](diagrams/full-candidate-sketch.png)

![Iterative inference pipeline from the side chat](diagrams/iterative_inference_pipeline.png)

[Scalable vector version of the iterative diagram](diagrams/iterative_inference_pipeline.svg)

The main chat also showed the same fragment under different category cues. The current version of that figure is below.

![Current baseline comparison of correct and misleading cues](outputs/baseline/cue_control.png)

## November 11, 2026 — What “label-free” leaves in place

**You asked** what a local geometric prior would learn and whether the competitive Hebbian model already learned from pixels.

**I did** compare the inputs of the two approaches. The implemented learner sees a row of supplied properties, such as ear shape and nose shape. The proposed geometric learner would see local edge measurements and their positions.

**What needed an ELI5 explanation.** Imagine two short pencil marks separated by a gap. Their directions and relative positions can make a smooth continuation plausible before you know what object they belong to. Learning that relationship would require the model to detect the marks and represent their relative geometry.

**Your new ideas and questions.** If neither approach requires a “cow” label, what would the proposed version add?

**My clarifications.** Category labels and feature descriptions are different kinds of supervision. The competitive learner can discover that supplied features tend to occur together. Learning where a feature begins and ends in the image is a further problem.

## November 12, 2026 — Associative memory and factor recovery

**You asked** us to keep the terms *renderer* and *generator* consistent, explain supervised Hebbian learning, and choose between implementing pixel-based geometry or a Hopfield exemplar prior. You chose Hopfield and asked where a resonator would fit in a future system.

**I did** add and execute the Hopfield notebook. It learns pairwise associations among feature values, scores complete recipes by energy, and separately tests retrieval from partial feature cues. I drew the three pipelines below. [Hopfield notebook](hopfield_learning.ipynb)

**What needed an ELI5 explanation.** In supervised Hebbian learning, a supplied label tells the model which unit is active while a feature appears. The co-activity rule still determines the weight change. In a Hopfield memory, combinations of features that fit learned associations can settle into low-energy states. A probability proportional to $e^{-\beta E}$ gives a lower-energy recipe more weight. Clamping a known feature is like holding one puzzle piece in place while trying the others.

**Your new ideas and questions.** Can a resonator recover shape, width and position from one bound vector? Can we tell whether an error came from encoding an image or from factorizing its code?

**My clarifications.** A resonator needs a suitable vector from an encoder. Testing it first on a known bound vector, then a complete image, then a masked image would separate those failure modes. Hopfield retrieval can settle on a wrong or novel combination. The diagram’s 17 units describe the original vocabulary. The current six three-way features use 18 units.

*Proposed encoder and resonator*

```mermaid
flowchart LR
    A["Visible pixels + mask"] --> B["Proposed image encoder"]
    B --> C["Noisy bound VSA vector"]
    C --> D["Resonator repeatedly updates factor estimates"]
    E["Codebook for each factor"] --> D
    D --> F["Proposed part, shape, width and position"]
    F --> G["Renderer draws the proposed completion"]
```

*Implemented supervised and competitive Hebbian priors*

```mermaid
flowchart TD
    A["Training drawing's six supplied feature values"]
    L["Supplied category label"] --> S["Activate the labeled unit"]
    A --> S
    A --> C["Anonymous units compete for activity"]
    S --> SW["Strengthen connections from observed features to that unit"]
    C --> CW["Move each unit's preferences toward the observed features in proportion to its activity"]
    SW --> P["Learned category prior over feature combinations"]
    CW --> P
    P --> I["Combine prior and image evidence for each candidate drawing"]
    V["Visible test pixels + mask"] --> E["Score agreement with each rendered candidate"]
    E --> I
    I --> O["Infer features and predict missing strokes"]
```

*Implemented Hopfield prior and partial-feature retrieval*

```mermaid
flowchart TD
    A["Complete training feature configurations"] --> B["Encode each as a 17-unit bipolar pattern"]
    B --> W["Learn pairwise connections from co-activity"]
    W --> P["Image experiment: compute each candidate's Hopfield energy"]
    P --> S["Use lower energy to give a candidate more prior weight"]
    V["Visible test pixels + mask"] --> E["Score agreement with rendered candidates"]
    E --> I["Combine image evidence and prior"]
    S --> I
    I --> O["Predict missing strokes"]
    W --> R["Feature retrieval experiment: update one free feature at a time"]
    Q["Directly supplied partial feature cue"] --> R
    R --> F["Keep known features fixed and accept energy-lowering changes"]
    F --> T["Return a completed feature configuration"]
```

## November 13, 2026 — Testing what a cue changes

**You asked** where the model used near-orthogonality in high dimensions, what exactly we would swap in a cue-conditioned encoder test, and whether the current experiment already did that comparison. You also asked for multiple seeds and visibilities, feature-specific cues beyond eye spacing, and a LaTeX writeup that began with definitions.

**I did** run 27 seed/face/visibility cases and 24 comparisons targeting nose, ears, head shape or layout. In those deliberately concealed cases, wrong hints changed the target in 6/6 nose, 6/6 ear, 6/6 head-shape and 4/6 layout comparisons. Revealing the distinguishing pixels reduced each count to 0/6. I wrote up the counted-prior experiment. The retained [writeup](writeup/closure_experiment.pdf) includes its later revisions.

**What needed an ELI5 explanation.** Nearly perpendicular random codes can be added without drowning one another out. That helps estimate pixel mismatch and read back feature scores. The proposed *swap* tests something else: save two interpretations of exactly the same strokes, one formed with a cow cue and one without. Feed each saved interpretation to the same neutral completion stage. If the output changes, the difference traveled through the saved encoding.

**Your new ideas and questions.** Could we compare a cue that only changes which recipes seem likely with one that changes how visible marks are grouped? If the encoder already expresses grouping, must the comparison still happen in pixels?

**My clarifications.** The existing model can change $p(z\mid c)$, the prior over recipe $z$ given cue $c$, while keeping its image-match score fixed. The proposed model adds a grouping that can itself depend on the cue. A renderer remains useful for checking a reconstruction against the marks, but an internal grouping comparison can also be measured directly. The swap below is an experimental design.

```mermaid
flowchart TD
    I["One image and one visibility mask"] --> F["Fixed filters detect visible strokes A and B"]
    F --> U["Group the visible strokes without a cue"]
    F --> C["Group the same visible strokes with feedback"]
    Q["Cow cue at the category layer"] --> C
    U --> E0["Save uncued grouping estimates"]
    C --> E1["Save cow-cued grouping estimates"]
    E0 --> S{"Choose one saved encoding per run"}
    E1 --> S
    S --> D["Same completion stage, downstream cue held neutral"]
    D --> R["Same renderer draws the inferred contours"]
```

## November 15, 2026 — Why give a pixel both a value and a role?

**You asked** what $r_{\rm pixel}$ contributes when $y_{uv}$ already says what is at row $u$, column $v$. You also asked whether zero is black or white, and why the code uses 0/1 when its vectors use −1/+1.

**I did** work through the encoding and the effect of changing conventions. [Image encoder](vsa_inference.py)

**What needed an ELI5 explanation.** In the drawings, 1 means a black stroke and 0 means white background. The pixel value says how much to add. Row and column codes say where. A role code says what kind of information is being stored. Multiplying every candidate and observation by the same bipolar role preserves their distances. With the same mask, switching pixel values to $2y-1$ doubles each difference and multiplies squared distance by four.

**Your new ideas and questions.** Would role labels become more valuable if an encoder stored features or stroke groups as well as pixels?

**My clarifications.** Distinct roles would let the same representation carry several kinds of information that could be queried separately. With the present single pixel role, one shared sign change per vector coordinate preserves the comparison. If the image convention changed, dividing the distance coefficient by four would preserve its contribution to the score.

## November 16, 2026 — Eye shapes, mixed hints and the poster

**You asked** to replace open/closed outline with three eye shapes, make dots characteristic of cows, oval pupils of cats and arches of humans, and rerun the comparisons. You also asked to mix correct and wrong feature hints, save the faces, and start a 24 × 36 inch poster with the cow, a neuroscience introduction and a simulation summary.

**I did** update the renderer to six three-way features, giving $3^6=729$ recipes. I reran 27 visibility cases and 30 targeted-cue cases. Wrong eye-shape hints changed the MAP eyes in 6/6 concealed cases and 0/6 revealed cases. I tested eight correct/wrong patterns across ear, eye and nose hints plus an uncued baseline. The saved output includes both MAP and posterior-mean faces at five weights. I also built the poster introduction and discussed the feedback interpretation of the drawing study you identified. [Targeted cues](outputs/targeted_feature_cues_2026-11-16/targeted_feature_cues.ipynb) · [Mixed cues](outputs/mixed_feature_cues_2026-11-16/mixed_feature_cues.ipynb) · [Visibility runs](outputs/seed_visibility_runs_2026-11-16/seed_visibility_results.ipynb)

**What needed an ELI5 explanation.** An earlier targeted trial gave one hint at a time. Saying “arched eyes” changes one feature’s score. Saying “arched eyes, long ears, muzzle” gives three pieces of evidence which may agree or conflict. A MAP face is one complete recipe. A gray area in the posterior-mean face shows where plausible recipes draw different strokes.

**Your new ideas and questions.** Would contradictory hints create more informative plots? Did the neuroscience paper identify a mechanism for the change in people’s drawings after disambiguation? What does scene context in a hidden region of V1 mean?

**My clarifications.** The V1 discussion concerned activity in the cortical region corresponding to a hidden part of the image. The paper discussed recurrent feedback as an explanation and used the full grayscale image for disambiguation. Its drawing result leaves open which computation groups the patches. The cartoon experiment supplies its own feature vocabulary.

![Results of concealing or revealing eye-shape evidence](outputs/targeted_feature_cues_2026-11-16/overview_eye_shape.png)

## November 17, 2026 — Giving people the same reconstruction task

**You asked** for contour plots of recovery against visibility, correct hints, noise and prior strength. You wanted a 50-trial interactive study that saved each six-feature response, feedback and a CSV. You requested adjustable semantic hints, disclosure of how many were right or wrong, a logarithmic visibility option and colors that reflected chance performance.

**I did** build the desktop slideshow, feature guide, session recorder and plotting tools. The model sweep compared each prior weight against zero on the same faces and fragments. At $\alpha=12$, the MAP recipe changed in 95% of matched cases at 10% requested visibility and 39% at 70%. Average recovery decreased at both endpoints. I updated the poster for the new feature set. [Study and analysis guide](psychophysicalExperiment/README.md) · [Prior-effect results](outputs/psychophysics/accuracy_contours/prior_effect.csv)

**What needed an ELI5 explanation.** Guessing one of three choices gives an average of $1/3$ correct per feature. Across six features, the expected score is $6/3=2$. An individual trial still has a whole-number score from zero to six. A higher prior weight can change the chosen face often while making it less accurate. “The prior mattered” and “the prior helped” require different measurements.

**Your new ideas and questions.** Could we give a participant anywhere from zero to six hints, tell them the total number correct, and leave them to decide which features to trust? Would stretching the low-visibility axis reveal changes concentrated there?

**My clarifications.** With a disclosed tally of $k$, a proposed face must match exactly $k$ hints under the study’s generation rule. With the tally hidden, each hint contributes according to its assumed reliability. Holding the number of supplied hints fixed makes the effect of hint correctness easier to interpret.

## November 18, 2026 — A usable desktop and browser study

**You asked** to use the full screen, enlarge feature choices, revise the instructions and display actual visibility. You wanted the study in its own folder and a browser version that saved feedback and responses locally.

**I did** revise the desktop layout and organize the study under `psychophysicalExperiment`. I published the static browser version after you made the repository public. At the time, I tested a complete 50-trial browser session, folder and ZIP saving, and Python replay of the export. [Browser files](psychophysicalExperiment/web/) · [Desktop launcher](<psychophysicalExperiment/Start face study.cmd>)

**What needed an ELI5 explanation.** No separate theory question was recorded. One interface distinction mattered: if the generator aims to reveal a percentage of blocks, the particular trial can reveal a different percentage of pixels. The number printed below the image tells the participant how much of *this face* is visible.

**Your new ideas and questions.** Could participants choose their own save directory at setup and take the study without installing Python?

**My clarifications.** Browsers that support directory access can update the CSV and save feedback after each confirmation. The other workflow downloads a session ZIP. Saving each trial’s mask, pixels and hints lets the model replay the stimulus the person actually saw.

## November 19, 2026 — The first 50-trial pilot

**You asked** for interpretable computer-result graphics, a clearer explanation of “zero correct hints,” more readable example faces, analysis of Gary’s pilot and a README that put the Bayesian question first. You also asked for a short account of why a flexible encoder remained the goal.

**I did** choose and plot several reconstruction sets. You selected set 06: at 37.5% realized visibility the model recovered 3, 4 and 6 features with 0, 3 and 6 correct hints. In the first pilot, 13/50 trials fell below 20% visibility and the minimum was 4.7%. I changed the default schedule to allocate 5, 10, 20 and 15 trials to progressively higher visibility bands, added a configurable noise cap, and updated the study and README. [Selected faces](outputs/poster/Face_Recovery_Sets/Set_06_38pct_visible/)

**What needed an ELI5 explanation.** The computer figure supplied six hints every time. At zero correct, all six were wrong. The 2/6 chance line is what uniform guessing earns on average. A model with visible marks, nonuniform priors or hints can score differently even when one source of information is weak.

**Your new ideas and questions.** Would a contour plot show the changing influence of hints more clearly than four nearly overlapping lines? The pilot raised a design question too: how many severely masked trials can a person answer meaningfully?

**My clarifications.** The figure used $\alpha=12$ for the prior and $\beta=1$ for hint strength. The pilot’s noise trend was irregular across one session. The next study schedule changed, so later pooling needs to preserve that difference. Hebbian and Hopfield models explain how prior preferences could be learned. The flexible-encoder question concerns how seeing a cue might change which image fragments count as one part.

## November 20, 2026 — Comparing learned priors with human recovery

**You asked** for Tom’s analyses, pooled anonymous human results, and comparable computer plots using label-free Hebbian learning. You requested matched comparisons with Hopfield, pixel flips capped at 20%, an extra high-visibility run, six visibility bands, trend captions, a Hebbian equation on the poster, and a revised writeup.

**I did** assemble the final poster and update the paper for the 729-recipe renderer. Three results answer different questions:

- **Same 840 trials, 5% flip probability.** At prior weight 12, counting and supervised Hebbian both averaged 4.63/6. Competitive Hebbian averaged 4.58/6 across three seeds. Kernel exemplar fitting scored 4.89/6 and Hopfield 5.45/6, or 5.01/6 after matching its effective prior entropy to the kernel. With prior weight zero, the same evidence and hints averaged 5.50/6. [Matched comparison](outputs/prior_comparison/summary.json)
- **Five flip rates, 64 held-out faces per condition.** With 2–20% pixel flips, label-free Hebbian averaged 4.19/6 across the initial 10–70% target visibility sweep. The separate 80%, 90% and 95% runs averaged 5.56/6 together. [Capped-noise results](outputs/capped_noise_contours/run_summary.json) · [High-visibility extension](outputs/capped_noise_high_visibility/run_summary.json)
- **Two 50-trial pilots.** The second participant averaged 4.58/6. Pooling both sessions gave 4.17/6. Under your poster rule, two human trials with six correct hints and less than 15% visibility were excluded, giving 4.13/6 over 98 trials. All 100 confirmed responses are retained. [Final human plot data](outputs/poster/plot_data/human_scatter_data.json)

**What needed an ELI5 explanation.** Why could two contours look different with the same dots? A measured trial is one point. Coloring the space between points requires a rule for connecting them. With sparse data, two rules can color the same gap differently. The revised plot marks weakly supported regions. Fractional computer scores also have a simple origin: each face earns an integer from 0 to 6, then the figure averages across faces, hint conditions, training seeds or flip rates as applicable.

**Your new ideas and questions.** You wanted a fairer visual comparison across noise and visibility, questioned whether six disclosed correct hints made a low-visibility human trial trivial, and asked how flexible grouping could preserve the shape of an unusual contour. You chose the label-free prior for the poster’s computer panels.

**My clarifications.** Participants could use the hint-accuracy tally. The poster simulation received six hints with hidden tally and fixed reliability. The two human sessions also used different visibility schedules. Similar trends motivate a matched comparison, but these pilot averages alone cannot establish that people used the model’s update rule. The final paper keeps cue-dependent grouping as future work. [Poster](Poster/Bayesian_Visual_Contour_Poster_Hebbian_Rule.pptx) · [Writeup](writeup/closure_experiment.pdf)

The two contour images embedded in the final poster:

![Computer recovery by visibility and correct hints](outputs/poster/plot_data/computer_contour.png)

![Pooled pilot recovery by visibility and correct hints](outputs/poster/plot_data/human_contour.png)

## November 21, 2026 — Establishing BayesContourProcessing

**You asked** to consolidate both model folders, keep the latest results and poster, recover the chat diagrams, and start a daily log at the cow image. You then asked to merge the two model folders, group their outputs and publish the consolidated directory as the repository’s main branch.

**I did** create this working directory, recover the diagrams and original comparison sketches, collect the latest saved runs and writeup, and reunite Tom’s archived trial files with his analyses. The copy passed the baseline model’s 27 tests and the learning variant’s 66 tests. I merged the implementations at the project root, preserved both full notebooks and their benchmark results, and moved the study sessions and poster plot data into [outputs](outputs/). [Project index](README.md)

All eight notebooks executed in an isolated copy, and all 66 distinct tests passed. The merged code matched the original code on all 1,107 rows of each model’s benchmark. The archived baseline CSV used the earlier dot-eye geometry and was reproducible with renderer version 1. The study replayed all 107 confirmed responses across the four saved sessions.

I connected this working directory to the existing GitHub repository and based its main branch on the previous main commit. The existing ignore rules keep raw participant sessions and the two full-notebook builders local.

**What needed an ELI5 explanation.** No new scientific explanation was requested today.

**Your new ideas and questions.** You designated this directory as the home for future project work and asked that the record reflect how your questions and understanding changed over time.

**My clarifications.** A repeated export is one session. Future entries belong under their Pacific calendar date, with additional work on the same day added to that day’s section.

## November 22, 2026 — Showing what a hint fixes

**You asked** for two faces under “Hint Tuning” in your revised 24 × 36 inch poster. One would show triangle ears in black and the other a tall face outline in black. You wanted the uncertain features in grayscale and both images sized to match the recovery candidates.

**I did** add the two equal-sized faces to a copy of your edited poster. Each image averages the 243 face combinations that satisfy its hint. I checked that both fit between the “Adjust” and “Infer” arrows. [Updated poster](Poster/Bayesian_Visual_Contour_Poster_24x36_Hint_Tuning.pptx)

I exported the [poster PNG](Poster/Bayesian_Visual_Contour_Poster_24x36_Hint_Tuning.png) at 200 dpi and added a Git ignore rule for new files in `Poster/Written_parts/`.

**What needed an ELI5 explanation.** Imagine stacking every drawing with triangle ears. The ears line up in every drawing, so they stay black. The eyes, nose, and other strokes move around. Where more drawings put ink in the same spot, the gray is darker. The tall-face example fixes the outline and varies the other features the same way.

**Your new ideas and questions.** You used the two hints to illustrate how a cue can specify one part while leaving the rest of the face open.

**My clarifications.** Gray shows how often a pixel appears when the compatible recipes are weighted equally. These are illustrations of hint information before the model scores the visible image or applies learned familiarity.

### Weighting the visual evidence

**Research question.** How much evidence for a face does a difference in encoded image distance provide? The revised writeup introduces a weight on the VSA score with the learned prior and hint reliability held fixed. The reported runs used visual weight one. The earlier prior-weight sweep, including α = 12, measures sensitivity to learned familiarity. [Writeup](writeup/closure_experiment.pdf)

**Completed comparison.** On the same 12 held-out faces, image evidence alone recovered 5.35 of six features on average. Adding hints at 50% assumed reliability gave 5.55, with exact recovery rising from 55.8% to 69.8%. The sweep varied the prior weight and assumed hint reliability across ten visibility levels and zero through six correct hints. It therefore answers a different question from fitting a visual weight. [Sweep](sweep_hint_reliability.py) · [Summary](outputs/hint_reliability/summary.csv) · [Condition results](outputs/hint_reliability/by_condition.csv)

**What needed an ELI5 explanation.** Flattening an image keeps all its pixels. Counting disagreements summarizes one candidate comparison exactly under the independent flip model. Superposing pixels into 512 code coordinates adds cross-talk. Across random codebooks the squared distance has the right mean, while a particular pair of faces can look closer or farther apart.

**New idea and clarification.** The exact encoded likelihood sums the probabilities of visible images that produce the observed code. VSA retrieval theory and compressive imaging give related analytic results under their own noise assumptions. A proposed experiment would compare visual weights across code dimensions on separate calibration and evaluation images. The compression ratio alone does not determine the weight. The poster is deferred.

### Poster results and publication

The revised poster pairs the concealed and visible nose examples and reports exact face recovery with image evidence alone (56%) and with hints that were correct half the time (70%). The visibility-band and noise-rate plots now combine computer and human results. Orange and teal circles show computer condition means and human trials, respectively. Larger triangles show group means, and dashed quadratic curves summarize their trends. Both plots have boxed legends and align beneath the two contour plots.

At your request, I exported the current 24 × 36 inch poster at 200 dpi (4,800 × 7,200 pixels) and made the PNG the primary README poster link. This update presents existing results. The discussion of VSA distance and likelihood scaling remains a separate writing task. [Poster PNG](Poster/Bayesian_Visual_Contour_Poster_24x36_Hint_Results_Merged_Plots_Fitted.png) · [Editable PowerPoint](Poster/Bayesian_Visual_Contour_Poster_24x36_Hint_Results_Merged_Plots_Fitted.pptx)
