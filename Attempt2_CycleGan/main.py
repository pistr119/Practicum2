 
import tensorflow as tf
 
import numpy as np
#from scipy.misc import imsave
from imageio import imwrite
import os
import shutil
from PIL import Image
import time
import random
import sys


from layers import *
from model import *

img_height = 256
img_width = 256
img_layer = 3
img_size = img_height * img_width

#to_train = True
#t#Zo_test = False
to_restore = True
output_path = "./output"
check_dir = "./output/checkpoints/"
 
 
temp_check = 0




max_epoch = 50000
max_images = 100

h1_size = 150
h2_size = 300
z_size = 100
batch_size = 1
pool_size = 50
sample_size = 10
save_training_images = True
ngf = 32
ndf = 64

class CycleGAN():
    def input_setup(self):
        '''
            taking image input          
        '''
        directory_A="/home/datadude/ganlab/data/vangogh2photo/trainA/*.jpg"
        directory_B="/home/datadude/ganlab/data/vangogh2photo/trainB/*.jpg"
        #filenames_A=tf.train.match_filenames_once()
        filenames_A=tf.train.match_filenames_once(directory_A)
        self.queue_length_A=tf.size(filenames_A)
        
        filenames_B = tf.train.match_filenames_once(directory_B)    
        self.queue_length_B = tf.size(filenames_B)
          
        filename_queue_A = tf.train.string_input_producer(filenames_A)
        filename_queue_B = tf.train.string_input_producer(filenames_B)
        
        image_reader = tf.WholeFileReader()
        _, image_file_A = image_reader.read(filename_queue_A)
        _, image_file_B = image_reader.read(filename_queue_B)       
        
        self.image_A = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_A),[img_height,img_height]),127.5),1)
        self.image_B = tf.subtract(tf.div(tf.image.resize_images(tf.image.decode_jpeg(image_file_B),[img_height,img_height]),127.5),1)


    def input_read(self, sess):
          '''
          reads input from image to folder.
          '''
          # Loading images into the tensors
          
          coord = tf.train.Coordinator()
          threads = tf.train.start_queue_runners(coord=coord)

          num_files_A = sess.run(self.queue_length_A) #length from input setup
          num_files_B = sess.run(self.queue_length_B)

          self.fake_images_A = np.zeros((pool_size,1,img_height, img_width, img_layer))
          self.fake_images_B = np.zeros((pool_size,1,img_height, img_width, img_layer))                
        
          self.A_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))
          self.B_input = np.zeros((max_images, batch_size, img_height, img_width, img_layer))# initialize
          print(max_images)
          for i in range(max_images): 
              image_tensor = sess.run(self.image_A)
              if(image_tensor.size == img_size*batch_size*img_layer):
              #if(image_tensor.size() == 199608): # ((256*256)*1*3)):
                  self.A_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

          for i in range(max_images):
              image_tensor = sess.run(self.image_B)
              if(image_tensor.size == img_size*batch_size*img_layer):
                  self.B_input[i] = image_tensor.reshape((batch_size,img_height, img_width, img_layer))

          coord.request_stop()
          coord.join(threads)      
        
    def model_setup(self):
        '''
            this function sets up model to train
        '''        
        self.input_A=tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_A")
        self.input_B=tf.placeholder(tf.float32, [batch_size, img_width, img_height, img_layer], name="input_B")
        
        self.fake_pool_A=tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_A")
        self.fake_pool_B=tf.placeholder(tf.float32, [None, img_width, img_height, img_layer], name="fake_pool_B")
        
        self.global_step=tf.Variable(0, name="global_step", trainable=False)
        
        self.num_fake_inputs=0
        
        self.lr=tf.placeholder(tf.float32, shape=[], name="lr")
        
        with tf.variable_scope("Model") as scope:
            self.fake_A=build_generator_resnet_9blocks(self.input_A, name="g_A")
            self.fake_B = build_generator_resnet_9blocks(self.input_B, name="g_B")
            
            self.rec_A = build_gen_discriminator(self.input_A, "d_A")
            self.rec_B = build_gen_discriminator(self.input_B, "d_B")
            
            scope.reuse_variables()
            
            self.fake_rec_A = build_gen_discriminator(self.fake_A, "d_A")
            self.fake_rec_B = build_gen_discriminator(self.fake_B, "d_B")
            
            self.cyc_A = build_generator_resnet_9blocks(self.fake_B, "g_B")
            self.cyc_B = build_generator_resnet_9blocks(self.fake_A, "g_A")
            
            scope.reuse_variables()

            self.fake_pool_rec_A = build_gen_discriminator(self.fake_pool_A, "d_A")
            self.fake_pool_rec_B = build_gen_discriminator(self.fake_pool_B, "d_B")
            
    def loss_calc(self):

        ''' var for loss calcs
        '''

        cyc_loss = tf.reduce_mean(tf.abs(self.input_A-self.cyc_A)) + tf.reduce_mean(tf.abs(self.input_B-self.cyc_B))
        
        disc_loss_A = tf.reduce_mean(tf.squared_difference(self.fake_rec_A,1))
        disc_loss_B = tf.reduce_mean(tf.squared_difference(self.fake_rec_B,1))
        
        g_loss_A = cyc_loss*10 + disc_loss_B
        g_loss_B = cyc_loss*10 + disc_loss_A

        d_loss_A = (tf.reduce_mean(tf.square(self.fake_pool_rec_A)) + tf.reduce_mean(tf.squared_difference(self.rec_A,1)))/2.0
        d_loss_B = (tf.reduce_mean(tf.square(self.fake_pool_rec_B)) + tf.reduce_mean(tf.squared_difference(self.rec_B,1)))/2.0

        
        optimizer = tf.train.AdamOptimizer(self.lr, beta1=0.5)

        self.model_vars = tf.trainable_variables()

        d_A_vars = [var for var in self.model_vars if 'd_A' in var.name]
        g_A_vars = [var for var in self.model_vars if 'g_A' in var.name]
        d_B_vars = [var for var in self.model_vars if 'd_B' in var.name]
        g_B_vars = [var for var in self.model_vars if 'g_B' in var.name]
        
        self.d_A_trainer = optimizer.minimize(d_loss_A, var_list=d_A_vars)
        self.d_B_trainer = optimizer.minimize(d_loss_B, var_list=d_B_vars)
        self.g_A_trainer = optimizer.minimize(g_loss_A, var_list=g_A_vars)
        self.g_B_trainer = optimizer.minimize(g_loss_B, var_list=g_B_vars)

        for var in self.model_vars: 
            print(var.name)

        #Summary variables for tensorboard

        self.g_A_loss_summ = tf.summary.scalar("g_A_loss", g_loss_A)
        self.g_B_loss_summ = tf.summary.scalar("g_B_loss", g_loss_B)
        self.d_A_loss_summ = tf.summary.scalar("d_A_loss", d_loss_A)
        self.d_B_loss_summ = tf.summary.scalar("d_B_loss", d_loss_B)

    def save_training_images(self, sess, epoch):
        
        #make output dirs
        if not os.path.exists("./output/imgs"):
            os.makedirs("./output/imgs")
        #save 10 images in every epoch. 
        for i in range(0,10):
            fake_A_temp, fake_B_temp, cyc_A_temp, cyc_B_temp= sess.run([self.fake_A, self.fake_B, self.cyc_A, self.cyc_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
            imwrite("./output/imgs/fakeB_"+ str(epoch) + "_" + str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
            imwrite("./output/imgs/fakeA_"+ str(epoch) + "_" + str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
            imwrite("./output/imgs/cycA_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_A_temp[0]+1)*127.5).astype(np.uint8))
            imwrite("./output/imgs/cycB_"+ str(epoch) + "_" + str(i)+".jpg",((cyc_B_temp[0]+1)*127.5).astype(np.uint8))
            imwrite("./output/imgs/inputA_"+ str(epoch) + "_" + str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
            imwrite("./output/imgs/inputB_"+ str(epoch) + "_" + str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))           

    def fake_image_pool(self, num_fakes, fake, fake_pool):
            '''saves gen images to pool of images
            '''
            if(num_fakes<pool_size):
                fake_pool[num_fakes]=fake
                return fake
            else:
                p=random.random()
                if p>.5:
                    random_id=random.randint(0, pool_size-1)
                    temp=fake_pool[random_id]
                    fake_pool[random_id]=fake
                    return temp
                else:
                    return fake
                


    def train(self):
        '''training function 
        
        '''
        
        init=tf.initialize_all_variables()
        self.input_setup()
        
        self.model_setup()
        
        self.loss_calc()
        
        
        
        saver=tf.train.Saver()
        
        with tf.Session() as sess:
            #https://flonelin.wordpress.com/2017/07/03/attempting-to-use-uninitialized-value-matching_filenames/
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
           #init = tf.initialize_all_variables()
            sess.run(init)
 
            '''read input...'''
            self.input_read(sess)
            
            if to_restore:
                chkpt_fname=tf.train.latest_checkpoint(check_dir)
                saver.restore(sess, chkpt_fname)
                
            writer=tf.summary.FileWriter("./output/2")
            
            if not os.path.exists(check_dir):
                os.makedirs(check_dir)#create dir if not exists
            
            #training loop (aka meat and potatoes)
            print("starting training, take a nap or something")
            for epoch in range(sess.run(self.global_step), max_epoch):
                print ("In epoch " , epoch)
                saver.save(sess, os.path.join(check_dir, "cyclegan"), global_step=epoch)
                
                #dealing w/ lr per epoch
                
                if(epoch<100):
                    curr_lr=.0002
                else:
                    curr_lr=.0002 -.0002*(epoch-100)/100 #adjust lr each time.  .
                
                if(save_training_images):
                    self.save_training_images(sess,epoch)
                
                for ptr in range(0, max_images):
                    print("in iter", ptr)
                    print("start time " + (str)(time.time()*1000.00))
                    
                #oprimize gen network
                
                _, fake_B_temp, summary_str=sess.run([self.g_A_trainer, self.fake_B, self.g_A_loss_summ], feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                writer.add_summary(summary_str, epoch*max_images +ptr)

                fake_B_temp1=self.fake_image_pool(self.num_fake_inputs, fake_B_temp, self.fake_images_B)
                
                #optimize D_B net                
                _, summary_str = sess.run([self.d_B_trainer, self.d_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_B:fake_B_temp1})
                writer.add_summary(summary_str, epoch*max_images+ptr)
                
                #optimize G_B net
                _, fake_A_temp, summary_str = sess.run([self.g_B_trainer, self.fake_A, self.g_B_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr})
                writer.add_summary(summary_str, epoch*max_images + ptr)
                 
                fake_A_temp =self.fake_image_pool(self.num_fake_inputs, fake_A_temp, self.fake_images_A)
                
                #optimize D_A net
                _, summary_str = sess.run([self.d_A_trainer, self.d_A_loss_summ],feed_dict={self.input_A:self.A_input[ptr], self.input_B:self.B_input[ptr], self.lr:curr_lr, self.fake_pool_A:fake_A_temp})
                writer.add_summary(summary_str, epoch*max_images + ptr)

                self.num_fake_inputs+=1
                
            sess.run(tf.assign(self.global_step, epoch + 1))
        writer.add_graph(sess.graph)
        
    def test(self):
        '''
        test function
        '''
        print("Testing the results...finally")
        self.input_setup()
        
        self.model_setup()
        
        reuse=tf.AUTO_REUSE
        saver=tf.train.Saver()
        init=tf.global_variables_initializer()
        
        with tf.Session() as sess:
            #init = tf.initialize_all_variables()
                        #https://flonelin.wordpress.com/2017/07/03/attempting-to-use-uninitialized-value-matching_filenames/
            init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
            sess.run(init)
            self.input_read(sess)
            
            chkpt_fname=tf.train.latest_checkpoint(check_dir)
            saver.restore(sess, chkpt_fname)
            
            if not os.path.exists("./output/imgs/test/"):
                os.makedirs("./output/imgs/test/")
            
            for i in range(0,100):
                fake_A_temp, fake_B_temp = sess.run([self.fake_A, self.fake_B],feed_dict={self.input_A:self.A_input[i], self.input_B:self.B_input[i]})
                imwrite("./output/imgs/test/fakeB_"+str(i)+".jpg",((fake_A_temp[0]+1)*127.5).astype(np.uint8))
                imwrite("./output/imgs/test/fakeA_"+str(i)+".jpg",((fake_B_temp[0]+1)*127.5).astype(np.uint8))
                imwrite("./output/imgs/test/inputA_"+str(i)+".jpg",((self.A_input[i][0]+1)*127.5).astype(np.uint8))
                imwrite("./output/imgs/test/inputB_"+str(i)+".jpg",((self.B_input[i][0]+1)*127.5).astype(np.uint8))

def main():
    tf.reset_default_graph()    #https://stackoverflow.com/questions/47296969/valueerror-variable-rnn-basic-rnn-cell-kernel-already-exists-disallowed-did-y#47297097
    model=CycleGAN()
    #if to_train:
    #    model.train()
   # elif to_test:
   #     model.test()
 #   model.train()
    model.test()

if __name__=='__main__':
    main()
                 
                 