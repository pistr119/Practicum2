

# Purpose
For my practicum, I was seeking to explore Generative Adversarial Networks (GANs).  It was exploratory in nature, and my goal was to create a GAN which produced "interesting" images.  I started with a celebrity face generator, and later explored Cycle Gans.  


# What are GANS?

Two main components:  Generator Neural Network and Discriminator Neural Network
Generator takes a random input, and converts it to a sample of data. (Shaikh, 2017)
Discriminator takes input from the generator, and predicts whether or not it is real or fake.  (Shaikh, 2017)

# Part I:  Train Discriminator

Forward propagation only

![DiscriminatorDiagram](https://github.com/pistr119/Practicum2/blob/master/Images/DiscriminatorDiagram.jpg?raw=true)


# Part II: Train Generator and Freeze Discriminator

Forward and back propogation happens in Generator training

![DiscriminatorDiagram](https://github.com/pistr119/Practicum2/blob/master/Images/GeneratorDiagram.png?raw=true)

# Steps to a GAN project

Define the problem 
Define architecture
Multi layer perceptrons?  CNN? 
Generate fake inputs for generator, train discriminator on fake data
Train generator with discriminator outputs
Train for a few epochs
Observe curve, and check manually

# Expected Challenges

Two networks dependent on each other
If one fails, everything fails
Instability:  often results in random oscillations
Weak gradient:  may be difficult to train when real and fake distributions are far from each other.  
Sensitive to hyperparameter selections
Mode collapse can be difficult to overcome

# My adventures
I started with trying to build a celebrity face generator.  
Used the ‘CELEBA” dataset: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
Initially, there was some success on small images (64x64).
While the algorithm knew “approximately” what a human face looked like, the faces did not look quite right.  


![initialfaces](https://github.com/pistr119/Practicum2/blob/master/Images/initialfaces.png?raw=true)
Graph started to look better, but actual improvement was minimal.
![lossesDecreasing](https://github.com/pistr119/Practicum2/blob/master/Images/lossesDecreasing.png?raw=true)

I decided to experiment with 128x128 images, rather than 64x64

![notSuccessfulGan128.png](https://github.com/pistr119/Practicum2/blob/master/Images/notSuccessfulGan128.png?raw=true)

No improvements despite training, and the generator never successfully generated images.  

![nosuccessfulImages.png](https://github.com/pistr119/Practicum2/blob/master/Images/nosuccessfulImages.png?raw=true)

Unfortunately, despite attempts, I did not make the 128x128pixel code work to generate actual faces.

Next, I experimented with CycleGan.  

![cycleGanDiagram.jpg](https://github.com/pistr119/Practicum2/blob/master/Images/cycleGanDiagram.jpg?raw=true)


# No luck....
While I got the code to work, despite 20k epochs of training, I couldn't get the GAN to work to produce images.  In the diagram above, the author converted horses to Zebras.  I tried to convert photographs to Van Gogh paintings, but was unsuccessful in doing so.  

![badcycleGan.jpg](https://github.com/pistr119/Practicum2/blob/master/Images/cycB_7_4.jpg?raw=true)
![badcycleGan.jpg](https://github.com/pistr119/Practicum2/blob/master/Images/cycB_7_5.jpg?raw=true)

# Conclusion
While the process was educational, it did not produce usable results.  A bit more experimentation would be needed in order to get a good GAN to work. 

# Other lessons learned
Always use Anaconda for environment installation.  Much easier to re-start, and easier to resolve dependencies. 
Compatibility in working with GPUs is hard.  One must have the right driver, cuda toolkit, and tensorflow/pytorch versions.  Any incompatiblities will result in CPU being used for training.  It's also very easy to install a package which downgrades PyTorch/Tensorflow GPU to regular packages. Always run tests to make sure that GPU is being used!


Generator/Discriminator diagrams from:  https://www.analyticsvidhya.com/blog/2017/06/introductory-generative-adversarial-networks-gans/
CycleGan code based on:  https://github.com/hardikbansal/CycleGAN/
CycleGan images from:  https://github.com/hardikbansal/CycleGAN/

Source of CycleGan dataset:  https://people.eecs.berkeley.edu/~taesung_park/CycleGAN/datasets/

