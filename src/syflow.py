import numpy as np
import torch
import flowtorch.bijectors as bij
import flowtorch.distributions as dist
import torch.nn as nn

from tqdm import tqdm

def flow_gen_default():
    return bij.Spline(count_bins=8)

'''Original implementation of Syflow as provided in the paper.'''
def syflow(X, Y, classifier,flow_population=None, subgroup_priors=[], pop_train_epochs=1000, subgroup_train_epochs=1000, final_fit_epochs=500,device=torch.device("cpu"),lr_flow=7e-3,lr_classifier=1e-5, verbose=False,batchsize=-1,
                   lambd=1,early_stopping_patience=50,feature_names=None,scaler=None, optimize_temperature=True,alpha=0.5,config=None, progressbar=True):
    
    torch.manual_seed(config.seed)
    if hasattr(config,"flow_gen"):
        flow_gen = config.flow_gen
    else:
        flow_gen = flow_gen_default
    X = X.to(device)
    Y = Y.to(device)
    if batchsize == -1:
        batchsize = X.shape[0]

    limits = torch.stack([torch.min(X,dim=0)[0],torch.max(X,dim=0)[0]],dim=1)
    limits.to(device)
    dim = Y.shape[1]

    classifier.to(device)
    
    if flow_population is None:
        bijector = flow_gen()
        base_dist = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim).to(device), torch.ones(dim).to(device)), 1)
        flow_population = dist.Flow(base_dist, bijector).to(device)
        pop_flow_optimizer = torch.optim.Adam(flow_population.parameters(), lr=lr_flow)
        
        print("Training population flow")
        best_loss = float('inf')
        epochs_without_improvement = 0

        for step in tqdm(range(pop_train_epochs), disable=not progressbar):
            idx = torch.randperm(X.shape[0], device=device)
            Y_batch = Y[idx]            
            pop_flow_optimizer.zero_grad()   
            log_p = flow_population.log_prob(Y_batch)
            
            if not torch.isfinite(log_p).all():
                print(f"Warning: Non-finite log_prob detected at step {step}, stopping population flow training")
                break
            
            log_p = -torch.mean(log_p)
            if not torch.isfinite(log_p):
                print(f"Warning: Non-finite loss at step {step}, stopping population flow training")
                break
            
            log_p.backward()
            torch.nn.utils.clip_grad_norm_(flow_population.parameters(), max_norm=1.0)
            pop_flow_optimizer.step()

            if log_p < best_loss:
                best_loss = log_p
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1

            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {step + 1} with loss {log_p:.4f}")
                break
    else:
        flow_population = flow_population.to(device)    
    
    bijector = flow_gen()
    base_dist = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim).to(device), torch.ones(dim).to(device)), 1)
    flow_subgroup = dist.Flow(base_dist, bijector).to(device)
    subgroup_flow_optimizer = torch.optim.Adam(flow_subgroup.parameters(), lr=lr_flow)

    bijector = flow_gen()
    base_dist = torch.distributions.Independent(torch.distributions.Normal(torch.zeros(dim).to(device), torch.ones(dim).to(device)), 1)
    flow_complement_subgroup = dist.Flow(base_dist, bijector).to(device)
    complement_flow_optimizer = torch.optim.Adam(flow_complement_subgroup.parameters(), lr=lr_flow)

    flow_steps = 2
    classifier_steps = 2
    counter_1 = 0
    counter_2 = 0

    optimizer_classifier = torch.optim.Adam(classifier.parameters(), lr=lr_classifier)

    regularization_prior = 0
    log_likelihood_population = flow_population.log_prob(Y).detach().clone()
    pop_mean = torch.mean(log_likelihood_population)

    prior_likelihoods = []
    prior_means = []
    best_loss = torch.inf
    for i,prior in enumerate(subgroup_priors):
        prior_likelihoods.append(prior.log_prob(Y).detach())
        prior_means.append(torch.mean(prior_likelihoods[-1]))

    for step in tqdm(range(subgroup_train_epochs), disable=not progressbar):
        # random batches of size batchsize over entire dataset
        idx = torch.randperm(X.shape[0],device=device)
        for i in range(0,X.shape[0],batchsize):
            idx_batch = idx[i:i+batchsize]
            X_batch = X[idx_batch]
            Y_batch = Y[idx_batch]
            optimizer_classifier.zero_grad()
            subgroup_flow_optimizer.zero_grad()
            complement_flow_optimizer.zero_grad()

            # train classifier
            classlabel = classifier(X_batch)
            subgroup = torch.argmax(classlabel,dim=1)

            try:
                likelihood_subgroup = flow_subgroup.log_prob(Y_batch)
                likelihood_complement = flow_complement_subgroup.log_prob(Y_batch)
            except RuntimeError as err:
                if verbose:
                    print(f"Warning: Failed to evaluate complement log_prob at step {step}: {err}")
                subgroup_flow_optimizer.zero_grad(set_to_none=True)
                complement_flow_optimizer.zero_grad(set_to_none=True)
                continue
                            
            # log(sum e^LL(y|c)*p(c)) = log(sum(e^LL(y|c)*e^log(p(c)) = log(sum(e^(LL(y|c)+log(p(c)))))
            likelihood = torch.stack([likelihood_complement,likelihood_subgroup],dim=1)
            class_data_likelihood = likelihood + torch.log(classlabel+1e-5)
            joint_likelihood = torch.logsumexp(class_data_likelihood,dim=1)

            likelihood_loss = -joint_likelihood.mean()# + likelihood_scale

            if torch.isnan(likelihood_loss):
                print(likelihood_subgroup,classlabel)

            if counter_1 < flow_steps:
                #print gradient
                likelihood_loss.backward()
                torch.nn.utils.clip_grad_norm_(flow_subgroup.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(flow_complement_subgroup.parameters(), max_norm=1.0)
                subgroup_flow_optimizer.step()
                complement_flow_optimizer.step()
                counter_1 += 1
            else:
                regularization_prior = 0
                likelihood_subgroup_centered = likelihood_subgroup-torch.mean(likelihood_subgroup)                
                
                # maximize the minimum distance to a prior subgroup
                for i,pl in enumerate(prior_likelihoods):
                    pl_batch = pl[idx_batch]
                    pl_mean = prior_means[i]
                    kl_prior = torch.sum(classlabel[:,1])**(alpha-1)*classlabel[:,1] * ((pl_batch-pl_mean) - (likelihood_subgroup_centered))
                    regularization_prior += kl_prior                    
                if len(subgroup_priors) > 0:
                    regularization_prior = regularization_prior/len(subgroup_priors)
                    
                kl_population = torch.sum(classlabel[:,1])**(alpha-1)*classlabel[:,1] * ((log_likelihood_population[idx_batch]-pop_mean) - (likelihood_subgroup_centered))
                loss = kl_population + lambd*regularization_prior
                loss = loss.sum()
                loss.backward()

                if isinstance(regularization_prior,int):
                    regularization_prior = torch.tensor([regularization_prior])
                if loss - lambd*torch.sum(regularization_prior) < best_loss:
                    best_loss = loss - lambd*torch.sum(regularization_prior)
 
                factor = classifier.c/2000
                factor = max(1,factor)
                factor = min(10,factor)
                classifier.cut_points.grad.data *= classifier.c/2000
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), X.shape[1])
                optimizer_classifier.step()
                classifier.fix_parameters()
                counter_2 += 1
                if counter_2 == classifier_steps:
                    counter_1 = 0
                    counter_2 = 0
                if len(subgroup_priors) > 0:
                    regularization_prior = regularization_prior.sum()*lambd
                    regularization_prior = regularization_prior.item()             
                            
        if optimize_temperature:
            if step ==  subgroup_train_epochs//2:
                classifier.temperature = classifier.temperature/2
            elif step == 3*subgroup_train_epochs//4:
                classifier.temperature = classifier.temperature/2

        if step % 100 == 0 and step > 0 and verbose:
            try:
                print('step: {}, data fit loss: {}, KL: {}, regularizer: {}, rule: {}'.format(step, likelihood_loss.item(), loss.item(), regularization_prior,classifier.get_rules(limits,feature_names=feature_names,scaler=scaler,X=X)))
            except:
                print('step: {}, data fit loss: {}, KL: {}, regularizer: {}'.format(step, likelihood_loss.item(), loss.item(), regularization_prior))
            subgroup_sizes = []
            for i in range(2):
                subgroup_sizes.append(torch.sum(subgroup==i).item())
            print("Subgroup sizes: ",subgroup_sizes)

    # fit flow on only subgroup data
    logits = classifier(X)
    classlabel = logits
    subgroup = torch.argmax(classlabel,dim=1)
    Y_subgroup = Y[subgroup==1]
    for step in range(final_fit_epochs):
        log_likelihood = flow_subgroup.log_prob(Y_subgroup)
        log_likelihood = -torch.mean(log_likelihood)
        subgroup_flow_optimizer.zero_grad()
        log_likelihood.backward()
        subgroup_flow_optimizer.step()

    # optimize temperature
    temp_grid = [2**(-i) for i in range(1,11)]
    log_likelihood_population = flow_population.log_prob(Y).detach().clone()
    pop_mean = torch.mean(log_likelihood_population)
    likelihood_subgroup = flow_subgroup.log_prob(Y).detach().clone()

    if optimize_temperature:
        with torch.no_grad():
            best_loss = None
            for temp in temp_grid:
                classifier.temperature = temp
                classlabel = classifier(X)
                loss = torch.sum(classlabel[:,1])**(alpha-1)*classlabel[:,1] * ((log_likelihood_population-pop_mean)- (likelihood_subgroup-torch.mean(likelihood_subgroup)))
                loss = loss.mean()
                if best_loss == None or loss < best_loss:
                    best_loss = loss
                    best_temp = temp
            classifier.temperature = best_temp
    kl_population = torch.sum(classlabel[:,1])**(alpha-1)*classlabel[:,1] * ((log_likelihood_population-pop_mean) - (likelihood_subgroup-torch.mean(likelihood_subgroup)))
    return (flow_population, flow_subgroup), classifier, kl_population.sum().item()


    
class And_Finder(nn.Module):
    def __init__(self, cut_points, temperature=0.2,epsilon=1e-5,bin_deviation=0.20,use_weights=True):
        super().__init__()
        n_variables = cut_points.shape[0]
        self.cut_points = nn.Parameter(cut_points.clone().detach(), requires_grad=True)


        self.zero = nn.Parameter(torch.zeros([n_variables,1],dtype=torch.float64), requires_grad=False)
        self.epsilon = epsilon
        self.temperature = temperature
        D = cut_points.shape[1]
        if D != 2:
            raise ValueError("And finder only works for two given cutpoints per feature")
        self.fixed_weights = torch.reshape(torch.linspace(1.0, D + 1.0, D + 1, dtype=torch.float64), [D+1])
        #repeat fixed weights for each variable
        self.fixed_weights = nn.Parameter(self.fixed_weights.clone().detach(),requires_grad=False)

        initial_weights = torch.rand([n_variables,],dtype=torch.float64)
        initial_weights[:] = 1
        
        self.and_weights = nn.Parameter(initial_weights, requires_grad=use_weights)
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU()

        limits = cut_points.clone().detach() 
        bin_deviation = bin_deviation #+ 0.01*n_variables
        # scale by 10% of the range
        limits[:,0] = limits[:,0] - bin_deviation*(limits[:,1]-limits[:,0])
        limits[:,1] = limits[:,1] + bin_deviation*(limits[:,1]-limits[:,0])
        self.limits = nn.Parameter(limits,requires_grad=False)

    def forward(self, x):
        cut_points = self.cut_points
        b = torch.cumsum(torch.cat([self.zero,-cut_points], 1),1)
        # repeat x along new dimension for each fixed weight
        x = x.unsqueeze(2)
        x = x.repeat(1,1,self.fixed_weights.shape[0])
        weights = self.fixed_weights.repeat(x.shape[0],x.shape[1],1)
        h = x * weights
        # add b to the batch
        b = b.repeat(x.shape[0],1,1)
        h = h + b
        h = h / self.temperature
        bins = self.softmax(h)
        #print(res.shape,weights.shape,x.shape,b.shape,self.and_weights.shape)
        
        # harmonic mean
        # res = x.shape[1]/torch.sum(1/bins[:,:,1],dim=1)
        importance = self.relu(self.and_weights)# + 0.01
        c = ((1+self.epsilon)/(bins[:,:,1]+self.epsilon))@(importance)
        res = torch.sum(importance)/c
        self.c = torch.sum(c).item()/c.shape[0]
        #self.a = torch.sum(importance).item()/importance.shape[0]
        # scale grad by c 
        # geometric mean
        #res = torch.exp(torch.sum(torch.log(bins[:,:,1])/x.shape[1],dim=1))
        res = torch.stack([1-res,res],dim=1)
        return res
    

    def get_rules(self,data_limits, feature_names, scaler, X):
        X = scaler.inverse_transform(X)
        cut_points = self.cut_points.data
        #print("C",self.c,"And grad",torch.mean(scale*torch.abs(self.and_weights.grad.data)).item(),"Cut point grad",torch.mean(scale*torch.abs(self.cut_points.grad.data)).item())
        if feature_names is None:
            feature_names = [f"X_{i}" for i in range(cut_points.shape[0])]
        if scaler is not None:
            cut_points = scaler.inverse_transform(cut_points.detach().cpu().numpy().T).T
            data_limits = scaler.inverse_transform(data_limits.detach().cpu().numpy().T).T
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()
        
        rule = []
        for i in range(cut_points.shape[0]):
            lower_bound, upper_bound = cut_points[i,:]
            if lower_bound < data_limits[i,0] and upper_bound > data_limits[i,1]:
                continue
            and_weight = self.and_weights[i]
            lower_bound = np.max([data_limits[i,0],lower_bound])
            upper_bound = np.min([data_limits[i,1],upper_bound])
            if and_weight < 0.1:
                continue
        
            nel = np.round(np.unique(X[:,i]))
            if len(nel) == 2 and 0 in nel and 1 in nel:
                if upper_bound < 1:
                    rule.append("NOT "+feature_names[i])
                else:
                    rule.append(feature_names[i])
            else:
                in_range = np.logical_and(X[:,i] >= lower_bound, X[:,i] <= upper_bound)
                inside = np.unique(X[in_range,i])
                out_range = np.logical_or(X[:,i] < lower_bound, X[:,i] > upper_bound)
                outside = np.unique(X[out_range,i])
                
                if len(inside) == 0:
                    # sometimes Syflow makes the interval slightly too narrow to contain any data points,
                    # so we include the closest data point to the interval
                    closest_lower = np.argmin(np.abs(X[:,i]-lower_bound))
                    closest_upper = np.argmin(np.abs(X[:,i]-upper_bound))
                    if np.abs(X[closest_lower,i]-lower_bound) < np.abs(X[closest_upper,i]-upper_bound):
                        inside = [X[closest_lower,i]]
                    else:
                        inside = [X[closest_upper,i]]                    
                if len(inside) == 1:
                    rule.append(f"{feature_names[i]} = {inside[0]:.2f}")
                elif len(outside) == 1:
                    rule.append(f"{feature_names[i]} != {outside[0]:.2f}")
                else:
                    lower_bound = np.min(inside)
                    upper_bound = np.max(inside)
                    if np.any(outside < lower_bound):
                        rule.append(f"{feature_names[i]} >= {lower_bound:.2f}")
                    elif np.any(outside > upper_bound):
                        rule.append(f"{feature_names[i]} <= {upper_bound:.2f}")
        return " AND ".join(rule)

    def get_utilized_features(self,data_limits,feature_names=None,scaler=None):
        cut_points = self.cut_points.data
        if feature_names is None:
            feature_names = [f"X_{i}" for i in range(cut_points.shape[0])]
        if scaler is not None:
            cut_points = scaler.inverse_transform(cut_points.detach().cpu().numpy().T).T
            data_limits = scaler.inverse_transform(data_limits.detach().cpu().numpy().T).T
        else:
            cut_points = cut_points.detach().cpu().numpy()
            data_limits = data_limits.detach().cpu().numpy()
        
        used_features = []
        for i in range(cut_points.shape[0]):
            lower_bound, upper_bound = cut_points[i,:]
            if lower_bound < data_limits[i,0] and upper_bound > data_limits[i,1]:
                continue
            and_weight = self.and_weights[i]
            if and_weight < 0.1:
                continue
            used_features.append(i)
        return used_features
    
    def get_and_weights(self):
        return self.and_weights

    # update parameters to be between 0 and 1
    def fix_parameters(self):
        #self.and_weights.data = torch.clamp(self.and_weights.data,0,5)
        # sort cut points
        self.cut_points.data, _ = torch.sort(self.cut_points.data)
        for i in range(self.cut_points.shape[0]):
            limits = self.limits[i,:]
            self.cut_points.data[i,:] = torch.maximum(self.cut_points.data[i,:],limits[0])
            self.cut_points.data[i,:] = torch.minimum(self.cut_points.data[i,:],limits[1])
