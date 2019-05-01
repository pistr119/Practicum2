# Purpose
For my practicum, I was seeking to explore Generative Adversarial Networks (GANs).  It was exploratory in nature, and my goal was to create a GAN which produced "interesting" images.  I started with a celebrity face generator, and later explored Cycle Gans.  


# What are GANS?

Two main components:  Generator Neural Network and Discriminator Neural Network
Generator takes a random input, and converts it to a sample of data. (Shaikh, 2017)
Discriminator takes input from the generator, and predicts whether or not it is real or fake.  (Shaikh, 2017)

# Part I:  Train Discriminator

Forward propagation only
![DiscriminatorDiagram](/images/DiscriminatorDiagram.jpg)