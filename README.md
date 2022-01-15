# open-sdf-net

This is a neural network approximating the truncated signed distance function of open curves.

### Intro

### Requirements

* PyTorch
* Tensorboard
* OpenCV
* Skimage - `conda install -c anaconda scikit-image`

### Get Started

* Modify `/configs/config.json`.
    * `batch size`, `learning rate`, `number of epochs` etc. are specific here.
    * `m`, `n` and `var` are related to the sampling process. 


1. Run `/code/drawer.py`.
    * Enter the name of your curve.
    * Use the mouse to draw the curve: 
      * Left click to add a vertex; 
      * Right click to finish drawing; 
      * Press any key to save the curve to `/curves`.

    
2. Run `/code/sampler.py`.
   * Enter the name of your curve.
   * This process may take a while as it generates a ground truth SDF heatmap in `/results`. 


3. Run `/code/trainer.py`.
    * Enter the name of your curve.
    * This process generates a predicted SDF heatmap in `/results`.
    

#### Optional
* Run `/code/renderer.py`.
  * Re-renderer the heatmap. 
* Run `/code/transformer.py`.
  * Regenerate the medial axis transform. 
* Run `/code/saver.py`.
  * Save all data to `/achived` with a timestamp. 