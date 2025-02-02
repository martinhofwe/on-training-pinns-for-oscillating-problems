"""
********************************************************************************
Author: Shota DEGUCHI
        Yosuke SHIBATA
        Structural Analysis Laboratory, Kyushu University (Jul. 19th, 2021)
implementation of PINN - Physics-Informed Neural Network in TensorFlow 2
Adapted by: Martin Hofmann-Wellenhof
********************************************************************************
"""

import os
import time
import datetime
import numpy as np
import tensorflow as tf

class PINN:
    def __init__(self, 
                t_ini, x_ini, y_ini, u_ini, 
                t_bndx, x_bndx, y_bndx, 
                t_bndy, x_bndy, y_bndy, 
                t_pde, x_pde, y_pde, 
                f_in, f_out, width, depth, TX, u_FDM, test_time_idx,
                w_init = "Glorot", b_init = "zeros", act = "tanh", 
                lr = 1e-3, opt = "Adam", 
                f_scl = "minmax", laaf = False, c = 1., 
                w_ini = 1., w_bnd = 1., w_pde = 1., BC = "Neu", 
                f_mntr = 10, r_seed = 1234, out_path = ""):

        # configuration
        self.dat_typ = tf.float32
        self.f_in    = f_in
        self.f_out   = f_out
        self.width   = width
        self.depth   = depth
        self.w_init  = w_init
        self.b_init  = b_init
        self.act     = act
        self.lr      = lr
        self.opt     = opt
        self.f_scl   = f_scl
        self.laaf    = laaf
        self.c       = c
        self.w_ini   = w_ini
        self.w_bnd   = w_bnd
        self.w_pde   = w_pde
        self.BC      = BC
        self.f_mntr  = f_mntr
        self.r_seed  = r_seed
        #self.random_seed(self.r_seed)

        # dataset
        self.t_ini = t_ini; self.x_ini = x_ini; self.y_ini = y_ini; self.u_ini = u_ini
        self.t_bndx = t_bndx; self.x_bndx = x_bndx; self.y_bndx = y_bndx
        self.t_bndy = t_bndy; self.x_bndy = x_bndy; self.y_bndy = y_bndy
        self.t_pde = t_pde; self.x_pde = x_pde; self.y_pde = y_pde

        # bounds (for feature scaling)
        bounds  = tf.concat([x_pde, y_pde, t_pde], 1)
        self.lb = tf.cast(tf.reduce_min (bounds, axis = 0), self.dat_typ)
        self.ub = tf.cast(tf.reduce_max (bounds, axis = 0), self.dat_typ)
        self.mn = tf.cast(tf.reduce_mean(bounds, axis = 0), self.dat_typ)

        # build
        self.structure = [self.f_in] + (self.depth-1) * [self.width] + [self.f_out]
        self.weights, self.biases, self.alphas, self.params = self.dnn_init(self.structure)

        # system param
        self.c = tf.constant(self.c, dtype = self.dat_typ)

        # optimization
        self.optimizer = self.opt_(self.lr, self.opt)
        self.ep_log       = []
        self.loss_log     = []
        self.loss_ini_log = []
        self.loss_bnd_log = []
        self.loss_pde_log = []

        # params for test loss
        self.t_inf = np.unique(TX[:, 0:1])
        self.x_inf = np.unique(TX[:, 1:2])
        self.y_inf = np.unique(TX[:, 2:3])
        self.u_FDM = u_FDM
        self.test_time_idx = test_time_idx
        self.output_path = out_path

        self.loss_p_test     = []
        self.loss_d_test = []
        
        # Self adapting weights
        self.optimizer_sa_ini = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        self.optimizer_sa_bndx = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        self.optimizer_sa_bndy = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        self.optimizer_sa_pde = tf.keras.optimizers.Adam(lr = 0.005, beta_1=.99)
        
        self.weights_sa_ini =  tf.Variable(tf.random.uniform([self.t_ini.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_bndx =  tf.Variable(tf.random.uniform([self.t_bndx.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_bndy =  tf.Variable(tf.random.uniform([self.t_bndy.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_pde =  tf.Variable(tf.random.uniform([self.t_pde.shape[0], 1], dtype=self.dat_typ))
        
        self.weights_sa_ini_ones =  tf.Variable(tf.ones([self.t_ini.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_bndx_ones =  tf.Variable(tf.ones([self.t_bndx.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_bndy_ones =  tf.Variable(tf.ones([self.t_bndy.shape[0], 1], dtype=self.dat_typ))
        self.weights_sa_pde_ones =  tf.Variable(tf.ones([self.t_pde.shape[0], 1], dtype=self.dat_typ))
        

        print("\n************************************************************")
        print("****************     MAIN PROGRAM START     ****************")
        print("************************************************************")
        print(">>>>> start time:", datetime.datetime.now())
        print(">>>>> configuration;")
        print("         random seed  :", self.r_seed)
        print("         data type    :", self.dat_typ)
        print("         activation   :", self.act)
        print("         weight init  :", self.w_init)
        print("         bias   init  :", self.b_init)
        print("         learning rate:", self.lr)
        print("         optimizer    :", self.opt)
        print("         width        :", self.width)
        print("         depth        :", self.depth)
        print("         structure    :", self.structure)

    def random_seed(self, seed = 1234):
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        tf.random.set_seed(seed)

    def dnn_init(self, strc):
        weights = []
        biases  = []
        alphas  = []
        params  = []
        for d in range(0, self.depth):   # depth = self.depth
            w = self.weight_init(shape = [strc[d], strc[d + 1]], depth = d)
            b = self.bias_init  (shape = [      1, strc[d + 1]], depth = d)
            weights.append(w)
            biases .append(b)
            params .append(w)
            params .append(b)
            if self.laaf == True and d < self.depth - 1:
                a = tf.Variable(1., dtype = self.dat_typ, name = "a" + str(d))
                params.append(a)
            else:
                a = tf.constant(1., dtype = self.dat_typ)
            alphas .append(a)
        return weights, biases, alphas, params

    def weight_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.w_init == "Glorot":
            std = np.sqrt(2 / (in_dim + out_dim))
        elif self.w_init == "He":
            std = np.sqrt(2 / in_dim)
        elif self.w_init == "LeCun":
            std = np.sqrt(1 / in_dim)
        else:
            raise NotImplementedError(">>>>> weight_init")
        weight = tf.Variable(
            tf.random.truncated_normal(shape = [in_dim, out_dim], \
            mean = 0., stddev = std, dtype = self.dat_typ), \
            dtype = self.dat_typ, name = "w" + str(depth)
            )
        return weight

    def bias_init(self, shape, depth):
        in_dim  = shape[0]
        out_dim = shape[1]
        if self.b_init == "zeros":
            bias = tf.Variable(
                tf.zeros(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        elif self.b_init == "ones":
            bias = tf.Variable(
                tf.ones(shape = [in_dim, out_dim], dtype = self.dat_typ), \
                dtype = self.dat_typ, name = "b" + str(depth)
                )
        else:
            raise NotImplementedError(">>>>> bias_init")
        return bias

    def opt_(self, lr, opt):
        if opt == "SGD":
            optimizer = tf.keras.optimizers.SGD(
                learning_rate = lr, momentum = 0.0, nesterov = False
                )
        elif opt == "RMSprop":
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate = lr, rho = 0.9, momentum = 0.0, centered = False
                )
        elif opt == "Adam":
            optimizer = tf.keras.optimizers.Adam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999, amsgrad = False
                )
        elif opt == "Adamax":
            optimizer = tf.keras.optimizers.Adamax(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        elif opt == "Nadam":
            optimizer = tf.keras.optimizers.Nadam(
                learning_rate = lr, beta_1 = 0.9, beta_2 = 0.999
                )
        else:
            raise NotImplementedError(">>>>> opt_")
        return optimizer

    def forward_pass(self, x):
        # feature scaling
        if self.f_scl == "minmax":
            z = 2. * (x - self.lb) / (self.ub - self.lb) - 1.
        elif self.f_scl == "mean":
            z = (x - self.mn) / (self.ub - self.lb)
        else:
            raise NotImplementedError(">>>>> forward_pass (f_scl)")
        # forward pass
        for d in range(0, self.depth - 1):
            w = self.weights[d]
            b = self.biases [d]
            a = self.alphas [d]
            u = tf.add(tf.matmul(z, w), b)
            u = tf.multiply(a, u)
            if self.act == "tanh":
                z = tf.tanh(u)
            elif self.act == "sine_all":
                z = tf.math.sin(u)
            elif self.act == "sine":
                if d == 0:
                    z = tf.math.sin(u)
                else:
                    z = tf.tanh(u)
            elif self.act == "swish":
                z = tf.multiply(u, tf.sigmoid(u))
            elif self.act == "gelu":
                z = tf.multiply(u, tf.sigmoid(1.702 * u))
            elif self.act == "mish":
                z = tf.multiply(u, tf.tanh(tf.nn.softplus(u)))
            else:
                raise NotImplementedError(">>>>> forward_pass (act)")

        w = self.weights[-1]
        b = self.biases [-1]
        a = self.alphas [-1]
        u = tf.add(tf.matmul(z, w), b)
        u = tf.multiply(a, u)
        z = u   # identity mapping
        y = z
        return y

    def pde(self, t, x, y):
        t = tf.convert_to_tensor(t, dtype = self.dat_typ)
        x = tf.convert_to_tensor(x, dtype = self.dat_typ)
        y = tf.convert_to_tensor(y, dtype = self.dat_typ)
        with tf.GradientTape(persistent = True) as tp:
            tp.watch(t)
            tp.watch(x)
            tp.watch(y)
            u = self.forward_pass(tf.concat([t, x, y], 1))
            u_t = tp.gradient(u, t)
            u_x = tp.gradient(u, x)
            u_y = tp.gradient(u, y)
        u_tt = tp.gradient(u_t, t)
        u_xx = tp.gradient(u_x, x)
        u_yy = tp.gradient(u_y, y)
        del tp
        gv = u_tt - (self.c ** 2) * (u_xx + u_yy)
        return u, gv

    def loss_ini(self, t, x, y, u, weights_sa_ini):
        with tf.GradientTape(persistent=True) as tp:
            tp.watch(t)
            tp.watch(x)
            tp.watch(y)
            u_, gv_ = self.pde(t, x, y)
        u_t_ = tp.gradient(u_, t)
        del tp
        loss = tf.reduce_mean(tf.square(weights_sa_ini*(u - u_))) \
                + tf.reduce_mean(tf.square(u_t_))
            
        return loss

    def loss_bnd(self, t, x, y):
        if self.BC == "Dir":
            u_, _ = self.pde(t, x, y)
            loss = tf.reduce_mean(tf.square(u_))
        elif self.BC == "Neu":
            with tf.GradientTape(persistent = True) as tp:
                tp.watch(t)
                tp.watch(x)
                tp.watch(y)
                u_, _ = self.pde(t, x, y)
            u_x_ = tp.gradient(u_, x)
            del tp
            loss = tf.reduce_mean(tf.square(u_x_))
        else:
            raise NotImplementedError(">>>>> loss_bnd")
        return loss

    def loss_bndx(self, t, x, y, weights_sa_bndx):
        if self.BC == "Dir":
            u_, _ = self.pde(t, x, y)
            loss = tf.reduce_mean(tf.square(weights_sa_bndx*u_))
        elif self.BC == "Neu":
            with tf.GradientTape(persistent = True) as tp:
                tp.watch(t)
                tp.watch(x)
                tp.watch(y)
                u_, _ = self.pde(t, x, y)
            u_x_ = tp.gradient(u_, x)
            del tp
            loss = tf.reduce_mean(tf.square(weights_sa_bndx*u_x_))
        else:
            raise NotImplementedError(">>>>> loss_bnd")
        return loss

    def loss_bndy(self, t, x, y, weights_sa_bndy):
        if self.BC == "Dir":
            u_, _ = self.pde(t, x, y)
            loss = tf.reduce_mean(tf.square(weights_sa_bndy*u_))
        elif self.BC == "Neu":
            with tf.GradientTape(persistent = True) as tp:
                tp.watch(t)
                tp.watch(x)
                tp.watch(y)
                u_, _ = self.pde(t, x, y)
            u_y_ = tp.gradient(u_, y)
            del tp
            loss = tf.reduce_mean(tf.square(weights_sa_bndy*u_y_))
        else:
            raise NotImplementedError(">>>>> loss_bnd")
        return loss

    def loss_pde(self, t, x, y, weights_sa_pde):
        _, gv_ = self.pde(t, x, y)
        loss = tf.reduce_mean(tf.square(weights_sa_pde*gv_))
        return loss

    @tf.function
    def loss_glb(self, 
                t_ini, x_ini, y_ini, u_ini, 
                t_bndx, x_bndx, y_bndx, 
                t_bndy, x_bndy, y_bndy, 
                t_pde, x_pde, y_pde, weights_sa_ini, weights_sa_pde, weights_sa_bndx, weights_sa_bndy):
        loss_ini = self.loss_ini(t_ini, x_ini, y_ini, u_ini,weights_sa_ini)
        loss_bnd = self.loss_bndx(t_bndx, x_bndx, y_bndx, weights_sa_bndx) + self.loss_bndy(t_bndy, x_bndy, y_bndy, weights_sa_bndy)
        loss_pde = self.loss_pde(t_pde, x_pde, y_pde, weights_sa_pde)
        loss = self.w_ini * loss_ini + self.w_bnd * loss_bnd + self.w_pde * loss_pde
        return loss

    def loss_grad(self, 
                t_ini, x_ini, y_ini, u_ini, 
                t_bndx, x_bndx, y_bndx, 
                t_bndy, x_bndy, y_bndy, 
                t_pde, x_pde, y_pde):
        with tf.GradientTape(persistent=True) as tp:
            loss = self.loss_glb(t_ini, x_ini, y_ini, u_ini, 
                                t_bndx, x_bndx, y_bndx, 
                                t_bndy, x_bndy, y_bndy, 
                                t_pde, x_pde, y_pde, self.weights_sa_ini, self.weights_sa_pde, self.weights_sa_bndx, self.weights_sa_bndy)
        grad = tp.gradient(loss, self.params)
        grad_sa_ini = tp.gradient(loss, self.weights_sa_ini)
        grad_sa_pde = tp.gradient(loss, self.weights_sa_pde)
        grad_sa_bndx = tp.gradient(loss, self.weights_sa_bndx)
        grad_sa_bndy = tp.gradient(loss, self.weights_sa_bndy)
        del tp
        return loss, grad, grad_sa_ini, grad_sa_pde, grad_sa_bndx, grad_sa_bndy
    
    @tf.function
    def grad_desc(self, 
                t_ini, x_ini, y_ini, u_ini, 
                t_bndx, x_bndx, y_bndx, 
                t_bndy, x_bndy, y_bndy, 
                t_pde, x_pde, y_pde):
        loss, grad, grad_sa_ini, grad_sa_pde, grad_sa_bndx, grad_sa_bndy = self.loss_grad(t_ini, x_ini, y_ini, u_ini, 
                                    t_bndx, x_bndx, y_bndx, 
                                    t_bndy, x_bndy, y_bndy, 
                                    t_pde, x_pde, y_pde)
        self.optimizer.apply_gradients(zip(grad, self.params))
        self.optimizer_sa_ini.apply_gradients(zip([-grad_sa_ini], [self.weights_sa_ini]))
        self.optimizer_sa_pde.apply_gradients(zip([-grad_sa_pde], [self.weights_sa_pde]))
        self.optimizer_sa_bndx.apply_gradients(zip([-grad_sa_bndx], [self.weights_sa_bndx]))
        self.optimizer_sa_bndy.apply_gradients(zip([-grad_sa_bndy], [self.weights_sa_bndy]))
        return loss

    def train(self, 
            epoch = 10 ** 5, batch = 2 ** 6, tol = 1e-5): 
        print(">>>>> training setting;")
        print("         # of epoch     :", epoch)
        print("         batch size     :", batch)
        print("         convergence tol:", tol)
        
        t_ini = self.t_ini; x_ini = self.x_ini; y_ini = self.y_ini; u_ini = self.u_ini
        t_bndx = self.t_bndx; x_bndx = self.x_bndx; y_bndx = self.y_bndx
        t_bndy = self.t_bndy; x_bndy = self.x_bndy; y_bndy = self.y_bndy
        t_pde = self.t_pde; x_pde = self.x_pde; y_pde = self.y_pde
        
        t0 = time.time()
        for ep in range(epoch):
            es_pat      = 0
            es_crt      = 5
            min_loss    = 100.
            ep_loss     = 0.
            ep_loss_ini = 0.
            ep_loss_bnd = 0.
            ep_loss_pde = 0.
            
            # full-batch training
            if batch == 0:
                ep_loss = self.grad_desc(t_ini, x_ini, y_ini, u_ini, 
                                         t_bndx, x_bndx, y_bndx, 
                                         t_bndy, x_bndy, y_bndy, 
                                         t_pde, x_pde, y_pde)
                ep_loss_ini = self.loss_ini(t_ini, x_ini, y_ini, u_ini, self.weights_sa_ini_ones)
                ep_loss_bnd = self.loss_bndx(t_bndx, x_bndx, y_bndx, self.weights_sa_bndx_ones) \
                                + self.loss_bndy(t_bndy, x_bndy, y_bndy, self.weights_sa_bndy_ones)
                ep_loss_pde = self.loss_pde(t_pde, x_pde, y_pde, self.weights_sa_pde_ones)
            
            # mini-batch training
            else:
                bound_b = min(self.x_ini.shape[0], 
                                self.x_bndx.shape[0],
                                self.x_bndy.shape[0],
                                self.x_pde.shape[0])
                idx_b = np.random.permutation(bound_b)

                for idx in range(0, bound_b, batch):
                    # batch for initial condition
                    t_ini_b = tf.convert_to_tensor(t_ini.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    x_ini_b = tf.convert_to_tensor(x_ini.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    y_ini_b = tf.convert_to_tensor(y_ini.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    u_ini_b = tf.convert_to_tensor(u_ini.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    # batch for boudary condition
                    t_bndx_b = tf.convert_to_tensor(t_bndx.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    x_bndx_b = tf.convert_to_tensor(x_bndx.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    y_bndx_b = tf.convert_to_tensor(y_bndx.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    t_bndy_b = tf.convert_to_tensor(t_bndy.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    x_bndy_b = tf.convert_to_tensor(x_bndy.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    y_bndy_b = tf.convert_to_tensor(y_bndy.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    # batch for PDE residual
                    t_pde_b = tf.convert_to_tensor(t_pde.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    x_pde_b = tf.convert_to_tensor(x_pde.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    y_pde_b = tf.convert_to_tensor(y_pde.numpy()[idx_b[idx:idx+batch if idx+batch < bound_b else bound_b]], dtype = self.dat_typ)
                    # compute loss and perform gradient descent
                    loss_b = self.grad_desc(t_ini_b, x_ini_b, y_ini_b, u_ini_b, 
                                            t_bndx_b, x_bndx_b, y_bndx_b, 
                                            t_bndy_b, x_bndy_b, y_bndy_b, 
                                            t_pde_b, x_pde_b, y_pde_b)
                    loss_ini_b = self.loss_ini(t_ini_b, x_ini_b, y_ini_b, u_ini_b, self.weights_sa_ini_ones)
                    loss_bnd_b = self.loss_bndx(t_bndx_b, x_bndx_b, y_bndx_b, self.weights_sa_bndx_ones) \
                                + self.loss_bndy(t_bndy_b, x_bndy_b, y_bndy_b, self.weights_sa_bndy_ones) 
                    loss_pde_b = self.loss_pde(t_pde_b, x_pde_b, y_pde_b, self.weights_sa_pde_ones)
                    # per batch -> per epoch
                    ep_loss     += loss_b     / int(bound_b / batch)
                    ep_loss_ini += loss_ini_b / int(bound_b / batch)
                    ep_loss_bnd += loss_bnd_b / int(bound_b / batch)
                    ep_loss_pde += loss_pde_b / int(bound_b / batch)

            ##### log loss:
            self.ep_log.append(ep)
            self.loss_log.append(ep_loss.numpy())
            self.loss_ini_log.append(ep_loss_ini.numpy())
            self.loss_bnd_log.append(ep_loss_bnd.numpy())
            self.loss_pde_log.append(ep_loss_pde.numpy())
            ### log test loss

            test_error_data = 0.
            test_error_physics = 0.
            for tm in self.test_time_idx:
                tm = np.array([tm])

                x_inf, y_inf = np.meshgrid(self.x_inf, self.y_inf)
                x_inf, y_inf = x_inf.reshape(-1, 1), y_inf.reshape(-1, 1)
                t_inf = np.tile(self.t_inf.reshape(-1, 1), (1, x_inf.shape[0])).T[:, tm]

                u_hat, gv_hat = self.infer(t_inf, x_inf, y_inf)
                test_error_data += tf.reduce_mean(tf.square(self.u_FDM[tm[0],:,:] - tf.reshape(u_hat, shape = [self.u_FDM.shape[1], self.u_FDM.shape[2]])))
                test_error_physics += tf.reduce_mean(tf.square(gv_hat))
                
                if ep % 2_000 == 0:
                    np.save(self.output_path + "res_" + str(tm) + "_" + str(ep)+".npy", tf.reshape(u_hat, shape = [self.u_FDM.shape[1], self.u_FDM.shape[2]]).numpy())


            self.loss_d_test.append((test_error_data/len(self.test_time_idx)).numpy())
            self.loss_p_test.append((test_error_physics/len(self.test_time_idx)).numpy())

            if ep % self.f_mntr == 0:
                elps = time.time() - t0

                print("ep: %d, loss: %.3e, loss_ini: %.3e, loss_bnd: %.3e, loss_pde: %.3e, elps: %.3f" 
                    % (ep, ep_loss, ep_loss_ini, ep_loss_bnd, ep_loss_pde, elps))
                t0 = time.time()
            
                if ep_loss < min_loss:
                    es_pat = 0
                    min_loss = ep_loss
                else:
                    es_pat += 1
                    print(">>>>> observed loss increase, patience: %d" % es_pat)

            if ep_loss < tol:
                print(">>>>> program terminating with the loss converging to its tolerance.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
            elif es_crt < es_pat:
                print(">>>>> program terminating with early stopping triggered.")
                print("\n************************************************************")
                print("*****************     MAIN PROGRAM END     *****************")
                print("************************************************************")
                print(">>>>> end time:", datetime.datetime.now())
                break
            else:
                pass

        print("\n************************************************************")
        print("*****************     MAIN PROGRAM END     *****************")
        print("************************************************************")
        print(">>>>> end time:", datetime.datetime.now())
                
    def infer(self, t, x, y):
        u_, gv_ = self.pde(t, x, y)
        return u_, gv_
