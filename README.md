# TensorFlood

The dataset is large - approximately 250,000~ instances, 150,000 of which have no ships at all. Where there are ships (one or more, their masks can overlap), they are usually represented as very, very small objects. I present the case with a mask on.
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/4579b14a-7617-4c68-a4c8-56353edf3da1)
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/d50cc9d6-fea9-4a00-b357-0bf3b01c0c2e)

  
I decided to use the U-Net architecture, with a cross-entropy loss function, taking class weights of 0.0027 and 0.998, as this is the distribution of labeled and unlabeled pixels on the entire dataset. Another option for leveling this classification problem is to take and sample a subset of, say, 1 to 9 photos where there are no ships to photos where there are ships. But in this problem, I tried to simply weight the loss function by sampling from the general population, which can worsen the imbalance of the dataset.
I have used three various probabilistic sampling methods:

Simple choice without return:
samples['simple probabilistic'] = np.random.choice(train_inds_copy, size=n_samples, replace = False)
Bernoulli sampling:
samples['bernoulli'] = np.asarray(train_inds_copy)[np.random.rand(len(train_inds_copy))<=pi]
Systematic sampling:
samples['systematic'] = np.asarray(train_inds_copy)[np.ceil(np.arange(r,  len(train_inds_copy)-1, a)).astype(int)]

![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/2ea3f043-9dcf-45fd-b529-ddac8a746432)
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/02bddc47-e0d5-4c0d-a7e3-dd4c7bb5079c)

 
Although the mask is segmented more accurately in the lighter pictures, the model has a noticeable problem with Precision, and a good Recall in general. Many pictures are complex and cannot be fully represented by a light dataset of 2000 instances, so the model has a clear problem of undertraining.

Then I created bigger model with more residual connections and trained it on a subsample of 1000 examples, augmented by cropping and resizing the mask.
The results were more promising.

![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/544f5162-5d7e-4d38-9e4c-05bb6c890bfc)
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/b1f65bd2-e4cb-4469-8f81-049d9d1756a9)
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/a75506c7-08a0-4d8d-9e52-d87c20137949)
![image](https://github.com/Romanomelchenko2001/TensorFlood/assets/47889749/0d0f18ef-e34e-4ad7-996e-a6c6d0c1fc5d)



  
  

