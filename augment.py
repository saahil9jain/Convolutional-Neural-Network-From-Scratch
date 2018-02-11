    def augment_data(self,X,y):
        #get a random sample of ~10% of data each time for each augmentation
        augments = []
        y_new = []
        for i in range(int(len(y)/10)):
            
            index = np.random.randint(0,len(y))
            #rotation
            sample = X[:,index]
            sample = sample.reshape(32,32,3)
            augmented = scipy.misc.imrotate(sample,30)
            np.append(X,augmented)
            augmented = augmented.flatten()/255.0
            augments.append(augmented)
            y_new.append(y[index])
            
            #reflection horiz
            index = np.random.randint(0,len(y))
            sample = X[:,index]
            sample = sample.reshape(32,32,3)
            augmented = np.fliplr(sample)
            augmented = augmented.flatten()/255.0
            augments.append(augmented)
            y_new.append(y[index])

            #reflection vert
            index = np.random.randint(0,len(y))
            sample = X[:,index]
            sample = sample.reshape(32,32,3)
            augmented = np.flipud(sample)
            augmented = augmented.flatten()/255.0
            augments.append(augmented)
            augments.append(y[index])
            
            #added noise
            index = np.random.randint(0,len(y))
            sample = X[:,index]
            sample = sample.reshape(32,32,3)
            noise = np.random.normal(size = sample.shape)
            augmented = sample + noise
            augmented = augmented.flatten()/255.0
            augments.append(augmented)
            y_new.append(y[index])
            
            #translate
            index = np.random.randint(0,len(y))
            sample = X[:,index]
            sample = sample.reshape(32,32,3)
            augmented = scipy.ndimage.interpolation.shift(sample,.1)
            augmented = augmented.flatten() / 255.0
            augments.append(augmented)
            y_new.append(y[index])
        augment_arr = np.column_stack(augments)
        labels = np.array(y_new)
        
        return X.append(augment_arr),y.append(labels)