# Deep Learning
Custom deep learing model for Gachon Deep Learning Class Competition (2020.04 ~ 2020.05)   
Gachon Deep Learning Class Competition : https://www.kaggle.com/c/gachon-deeplearning   
   
Created by __Team I__ (Lee NamJun, Yeo JunKu, Cho SoYeong, Choi JiWon)

# Dataset Dataset Description
We will use a subset of Yoga-82 dataset: https://sites.google.com/view/yoga-82/   
_balancing, inverted, reclining, sitting, standing, and wheel._

# Goal of Competition
Through this competition,
we hope to improve our deep learning and PyTorch coding skills so that we can finally have a wide perspective on the deep learning model.

# Our Approach
The primary logic of our approach is find out the best combination of element of model (hyper-parameters : optimizers, batch size, ...) with less epoch, then train model with the combination with more epoch.    
So, we use __Tuner__ to find out the best combination of hyper parameters and optimizer.
   
   
We used the following parameters:
~~~
config = {
    "lr": [0.001, 0.005, 0.0005],
    "batch_size": [64, 32, 16, 8],
    "optimizer": ['SGD','Adagrad', 'Adam']
}
~~~
~~~
class Tuner:
    def __init__(self, model, train_data, test_data, config, epoch):
        self.model = model
        self.train_data = train_data
        self.test_data = test_data
        self.config = config
        self.epoch = epoch
        self.combinations = list(product(config["lr"], config["batch_size"], config["optimizer"]))
        self.best_combination = self.tune_parameters()
        
    # Find out the best combination of elements
    def tune_parameters(self):
        best_combination = {"combination": [], "acc": 0.0}
        best_loss = 1
        best_epoch = 0
        best_model = copy.deepcopy(self.model)
        
        while len(self.combinations) >= 1:
            combi = self.combinations.pop()
            compare_model, optimizer, train_loader, val_loader = self.make_model(combi)
            
            # Training model with a combination
            print('Training combination: ', combi, ' ...')
            best = 0
            for epoch in range(self.epoch):
                train(compare_model, train_loader, optimizer, epoch)
                test_loss, test_accuracy = evaluate(compare_model, val_loader)
                if test_accuracy > best:
                    best = test_accuracy
                    best_loss = test_loss
                    best_epoch = epoch
                print('[{}] Test Loss : {:.4f}, Accuracy : {:.4f}%'.format(epoch, test_loss, test_accuracy))

                # Stop epoch if it doesn't renew more than 5 epochs based on lowest loss
                if epoch > best_epoch+5 and test_loss > best_loss:
                    break

            # Deciding best model
            if best > best_combination["acc"]:
                best_combination["acc"] = best
                best_combination["combination"] = combi
                best_model = compare_model
        
        model = best_model
        torch.save(model.state_dict(), "./best_model.pth")
        return best_combination["combination"]
        
    # Make a set of model elements
    def make_model(self, combi):
        # Deep Copy original model to training to initialize weights at every combinations
        compare_model = copy.deepcopy(self.model)
        compare_model = compare_model.to(DEVICE)
        
        # Make Optimizer
        if(combi[2] == 'SGD'):
            optimizer = optim.SGD(compare_model.parameters(), lr=combi[0], momentum = 0.9, nesterov=True)
        elif(combi[2] == 'Adagrad'):
            optimizer = optim.Adagrad(compare_model.parameters(), lr=combi[0])
        elif(combi[2] == 'Adam'):
            optimizer = optim.Adam(compare_model.parameters(), lr=combi[0])
        
        # Make Loader
        train_loader = DataLoader(self.train_data, batch_size=combi[1], shuffle=True)
        val_loader = DataLoader(self.test_data, batch_size=combi[1], shuffle=False)
        
        return compare_model, optimizer, train_loader, val_loader
~~~
~~~
# Get the best combination of hyper paramers and optimizer
combi = Tuner(model, train_set, val_set, config, 10).best_combination
~~~
 
As a result, we increased the accuracy from 60% to 85% using Tuner.   
![image](https://user-images.githubusercontent.com/63892688/123669792-6f99f280-d877-11eb-98ea-fdd2a8b84d89.png)

