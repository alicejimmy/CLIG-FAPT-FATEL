# Collect Human-annotated Labels of UPLL Dataset
## CIFAR100_SM_500, CIFAR100_T_500
* This research collected human-annotated labels for CIFAR100_SM_500 and CIFAR100_T_500, and the results are stored in `small_mammals_result.csv` and `trees_result.csv`, respectively.
    * To protect the privacy of the annotators, we represent each annotator with a number from 0 to 10.
* Considering time and manpower constraints, this research randomly selected 500 training set images from CIFAR100_SM and CIFAR100_T for annotation.
    * CIFAR100_SM_500 includes five categories: hamster, mouse, rabbit, shrew, and squirrel.
    * CIFAR100_T_500 includes five categories: maple, oak, palm, pine, and willow.
* There are a total of 11 annotators, including the author.
    * Each annotator annotated 500 small mammal dataset images and 500 tree dataset images.
    * Each annotator received 30 minutes of educational training before annotating images. The training content was to familiarize annotators with the characteristics of each category of animals and trees.
    * All annotators were required to complete the classification of all images within two weeks (2024/2/21~2024/3/6).
* We used Google forms to collect the annotations of each annotator. Each form consisted of 500 choice questions, with each question providing a random image and five options for the annotator to select the species they believe the image represents. An example is shown below:<br>
  <img src="https://github.com/alicejimmy/CLIG_FAPT_FATEL/assets/71706978/b8b2e50a-9aa3-4a5e-9721-fbb298c4ef2e" width="600" height="370">
* To facilitate annotators' completion, each questionnaire was divided into five smaller questionnaires, each with 100 questions.
* For detailed instructions on creating questionnaires, please refer to: `CLIG-FAPT-FATEL/dataset_collection/create_forms/`

## CIFAR-10N
* `CIFAR-10_human.pt` contains the human-annotated results of CIFAR-10N.
* This research did not collect human-annotated labels for CIFAR10's UPLL dataset. We used the publicly available dataset CIFAR-10N as a reference for generating the CIFAR10 UPLL dataset.
* For more details on CIFAR-10N, please refer to: http://noisylabels.com/
