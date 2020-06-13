import pandas as pd
import sys

BATCH_SIZE = 32
LEARNING_RATE = 0.001


def get_data_and_split():
    df = pd.read_csv(sys.argv[1], index_col = 0)

    train_split = int(0.8 * len(df.index.values))
    
    x_train = df.iloc[0:train_split,0:-1]
    y_train = df.iloc[0:train_split,-1]
    x_test = df.iloc[train_split:,0:-1]
    y_test = df.iloc[train_split:,-1]

    return x_train,y_train,x_test,y_test

def get_model_parameters():

    f = open(sys.argv[2],"r")
    parameters_str = f.read()

    parameters = dict() #parameters of features
    degrees = dict()    #polynomial degrees of features

    for parameter_str in parameters_str.split(' + '):
        parameter = 1.0 #initial parameters for features is assigned 1
        if('*' in parameter_str and '^' in parameter_str):
            feature = parameter_str[parameter_str.index('*')+1:parameter_str.index('^')]
            degree = float(parameter_str[parameter_str.index('^')+1:])
            parameters[feature] = parameter
            degrees[feature] = degree
        elif('*' in parameter_str):
            feature = parameter_str[parameter_str.index('*')+1:]
            degree = 1
            parameters[feature] = parameter
            degrees[feature] = degree
        else:
            parameters["bias"] = 0.0 #a value with no predictors set up as a bias.
            

    return parameters,degrees

def predict(sample,parameters,degrees):
    predict = 0.0
    for predictor,parameter in parameters.items():
        if(predictor != 'bias'):
            predict += parameter * pow(float(sample[predictor]), degrees[predictor])
        else:
            predict += parameter
    
    return predict

def gradient_descent(mini_batch,target,predictor,beta,parameters,degrees):
    sum_part = 0.0 #sum part of gradient
    n = len(mini_batch.index.values)
    

    for row in mini_batch.index.values: #for every sample row calculating the sum part of gradient
        
        if(predictor != 'bias'):
            prediction = predict(mini_batch.loc[row,:],parameters,degrees)
            sum_part += (float(target.at[row]) - prediction) * (- pow((float(mini_batch.at[row,predictor])),degrees[predictor]))
        else:
            prediction = predict(mini_batch.loc[row,:],parameters,degrees)
            sum_part += (float(target.at[row]) - prediction) * -1
        
    gradient = 2/n * sum_part
    
    return gradient

def mse(parameters,degrees,x,y):

    sum_mse = 0.0
    for row in x.index.values: #for every sample row calculate (yi-y^i)^2
        prediction = 0.0
        for predictor,parameter in parameters.items():
            if(predictor != 'bias'):
                prediction += parameter * pow(float(x.at[row,predictor]), degrees[predictor])
            else:
                prediction += parameter
        sum_mse += pow((float(y.at[row]) - prediction),2)
    
    mse = sum_mse / len(x.index.values)

    return mse

def train(x_train, y_train, parameters, degrees):

    number_of_batches = int(len(x_train.index.values) / BATCH_SIZE) + (len(x_train.index.values) % BATCH_SIZE > 0)
    epoch = 1
    increased_epoch = 0
    last_mse = 0 
    
    
    while(True):
        
        batch_start = 0
        batch_end = BATCH_SIZE
        
        
        for batch in range(0,number_of_batches):
            
            for predictor,beta in parameters.items():
                gradient = gradient_descent(x_train.iloc[batch_start:batch_end],y_train.iloc[batch_start:batch_end],predictor,beta,parameters,degrees) #calculating the gradient for beta by using the mini batch
                if(gradient == 0.0):
                    print("0")
                parameters[predictor] += LEARNING_RATE * (-gradient) #moving towards the negative way of gradient
                

            batch_start += BATCH_SIZE

            if(batch == number_of_batches - 2):
                batch_end = len(x_train.index.values)
            else:
                batch_end += BATCH_SIZE
        
        
        mse_value = mse(parameters,degrees,x_test,y_test)
        if(mse_value >= last_mse):
            increased_epoch += 1

        last_mse = mse_value
        print("Epoch {} MSE: {}".format(epoch,mse_value))
        

        if(increased_epoch == 3):
            print("Final MSE: {}".format(mse_value))
            break
        
        epoch += 1

    return parameters


if __name__ == "__main__":

    if(len(sys.argv) == 3):
        x_train,y_train,x_test,y_test = get_data_and_split()
        parameters,degrees = get_model_parameters()
        model_parameters = train(x_train, y_train, parameters, degrees)
    else:
        print("You must give your dataset and model in order to start.")