# RNN Night Fever

**Does hippocampal replay emerge when recurrent neural networks go to sleep?**

<img src="docs/Brain_image.png" alt="RNN Night Fever" width="280" align="right">

A [BrainHack Donostia 2026](https://brainhack-donostia.github.io) project (3–6 November 2026), led by Alejandro Tabas (BCBL). The full project description is in [`docs/RNN_Night_Fever_BHD2026.docx`](docs/RNN_Night_Fever_BHD2026.docx).

To perceive a noisy world, the brain needs an internal model of how its inputs are structured, but that model is too complex to learn in one go. We will build three coupled recurrent networks (perception, a fast hippocampus and a slow, high-capacity cortex), train them only to perceive well, and then let them "sleep". Our aim is to determine whether hippocampus-to-cortex replay emerges on its own during sleep, and under which conditions it does.

Existing computational models of consolidation build replay in: the modeller decides that the hippocampus will reactivate its contents during sleep and that the cortex will learn from them. We test the converse: if a system is optimised only to perceive well while awake, and is given idle time while asleep, does replay emerge by itself? And if so, what does it replay? The generative model of the world is stored in the network *states*, not the weights. Every episode samples a new world, spread over several "days" separated by sleep, and the weights are trained across episodes with backpropagation through the whole episode, so the networks learn how to learn a new world.


## Team map

| Team | Owns | Starts from |
|---|---|---|
| **Team 1** — generative models
| **Team 2** — architecture
| **Team 3** — analysis & abstract


## License

[GNU GPL v3.0](LICENSE).
