import numpy as np
import tensorflow as tf
import time
from datetime import datetime
import os
import matplotlib.pyplot as plt
import pickle

from data_simulation import get_simulated_data_two_mass
from plot_two_mass import plot_solution, plot_loss

#np.random.seed(12345)
#tf.random.set_seed(12345)


class Logger(object):
    def __init__(self, save_loss_freq, print_freq=1_000):
        print("TensorFlow version: {}".format(tf.__version__))
        print("Eager execution: {}".format(tf.executing_eagerly()))
        print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

        self.start_time = time.time()
        self.save_loss_freq = save_loss_freq
        self.print_freq = print_freq
        self.loss_over_epoch = []

    def __get_elapsed(self):
        return datetime.fromtimestamp((time.time() - self.start_time)).strftime("%m/%d/%Y, %H:%M:%S")

    def __get_error_u(self):
        return self.error_fn()

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, pinn):
        print("\nTraining started")
        print("================")
        print(pinn.model.summary())

    def log_train_epoch(self, epoch, loss, log_data):
        if epoch % self.save_loss_freq == 0:
            data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
            self.loss_over_epoch.append([data_error_m1, data_error_m2, physics_error_m1, physics_error_m2])
        if epoch % self.print_freq == 0:
            if not epoch % self.save_loss_freq == 0:
                data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
            print(f"{'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  train loss = {loss.numpy():.4e}  data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e} ")

    def log_train_end(self, epoch, log_data):
        print("==================")
        data_error_m1, data_error_m2, physics_error_m1, physics_error_m2 = [m.numpy() for m in log_data]
        print(
            f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} data error m1= {data_error_m1:.4e}  data error m2= {data_error_m2:.4e} physics error m1 = {physics_error_m1:.4e}  physics error m2 = {physics_error_m2:.4e}  ")


class PhysicsInformedNN(object):
    def __init__(self, layers, h_activation_function, logger, simul_constants, domain, physics_scale, lr, data,
                 simul_results, storage_path):

        inputs, outputs = self.setup_layers(layers, h_activation_function)
        self.model = tf.keras.Model(inputs=inputs, outputs=outputs, name="va_pinn_model")

        self.storage_path = storage_path
        self.dtype = tf.float32
        m2, m1, c1, d1, c2, d2 = simul_constants
        scaling_factor = 1.0
        self.c1 = tf.constant(c1 / scaling_factor, dtype=self.dtype)
        self.d1 = tf.constant(d1 / scaling_factor, dtype=self.dtype)
        self.c2 = tf.constant(c2 / scaling_factor, dtype=self.dtype)
        self.d2 = tf.constant(d2 / scaling_factor, dtype=self.dtype)
        self.m1 = tf.constant(m1 / scaling_factor, dtype=self.dtype)
        self.m2 = tf.constant(m2 / scaling_factor, dtype=self.dtype)

        self.x_ic, self.y_lbl_ic, self.x_physics, self.input_all, self.y_lbl_all = data
        self.y_m2_simul, self.y_m2_dx_simul, self.y_m2_dx2_simul, self.y_m1_simul, self.y_m1_dx_simul, self.y_m1_dx2_simul, _, _, _ = simul_results
        self.scaling_factor = tf.constant(1.0, dtype=self.dtype)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.8)
        self.logger = logger
        self.physics_scale = physics_scale

    def setup_layers(self, layers, h_activation_function):
        inputs = tf.keras.Input(shape=(layers[0],))
        x = inputs
        print("Setting up layers: ")
        for count, width in enumerate(layers[1:-1]):
            if h_activation_function == "sine":
                print(width, ": sine af")
                x = tf.keras.layers.Dense(width, activation=tf.math.sin)(x)
            elif h_activation_function == "single-sine" and count == 0:
                print(width, ": sine af")
                x = tf.keras.layers.Dense(width, activation=tf.math.sin)(x)
            else:
                print(width, ": tanh af")
                x = tf.keras.layers.Dense(width, activation=tf.keras.activations.tanh)(x)
        print("Output layer:")
        print(layers[-1], ": no af")
        outputs = tf.keras.layers.Dense(layers[-1], activation=None)(x)

        return inputs, outputs

    def store_intermediate_result(self, epoch, pred_params):
        m1_loss, m2_loss, pred_y_m1, pred_y_m2 = pred_params
        y_pred = tf.concat((pred_y_m1, pred_y_m2), axis=1)
        with open(os.path.join(self.storage_path, "loss_epoch.pkl"), "wb") as fp:
            pickle.dump(self.logger.loss_over_epoch, fp)

        plot_loss(self.logger.loss_over_epoch, self.physics_scale, os.path.join(self.storage_path, "loss"), scaled=False)
        plot_loss(self.logger.loss_over_epoch, self.physics_scale, os.path.join(self.storage_path, "loss_scaled"), scaled=True)
        plot_solution(self.input_all, self.y_lbl_all[:, 1], self.y_lbl_all[:, 0], self.x_ic, self.y_lbl_ic[:, 1],
                      self.y_lbl_ic[:, 0], y_pred, os.path.join(self.storage_path, "plots/res_"), epoch)
        plt.close('all')
        np.save(os.path.join(self.storage_path, "plots/res_" + str(epoch)+".npy"), y_pred)

    # The actual PINN
    def f_model(self, x_):
        with tf.GradientTape() as tape:
            tape.watch(x_)
            y = self.pred_with_grad(x_)
            y_dx = y[:, 2:]
            y_dx2_all = tape.batch_jacobian(y_dx, x_)[:, :, 0]
        del tape

        y_m1 = tf.expand_dims(y[:, 0], -1)
        y_m2 = tf.expand_dims(y[:, 1], -1)
        y_m1_dx = tf.expand_dims(y_dx[..., 0], -1)
        y_m2_dx = tf.expand_dims(y_dx[..., 1], -1)
        y_m1_dx2 = tf.expand_dims(y_dx2_all[..., 0], -1)
        y_m2_dx2 = tf.expand_dims(y_dx2_all[..., 1], -1)

        m1_loss = ((self.c1 * (- y_m1) + self.d1 * (- y_m1_dx) + self.c2 * (y_m2 - y_m1) + self.d2 * (
                    y_m2_dx - y_m1_dx)) / self.m1) - (
                              y_m1_dx2 / self.scaling_factor) 
        m2_loss = (((-self.c2 * (y_m2 - y_m1) - self.d2 * (y_m2_dx - y_m1_dx)) / self.m2) - (
                    y_m2_dx2 / self.scaling_factor))

        return [m1_loss, m2_loss, y_m1, y_m2]

    def pred_with_grad(self, x_points):
        with tf.GradientTape() as t:
            t.watch(x_points)
            pred = self.model(x_points)
        dx = t.batch_jacobian(pred, x_points)[:, :, 0]
        y_pred_full = tf.concat((pred, dx), axis=1)
        return y_pred_full

    def calc_loss_ic(self):
        diff = self.y_lbl_ic - self.pred_with_grad(self.x_ic)
        diff_m1 = tf.square(diff[:, 0:1])
        diff_m1_dx = 0.0
        diff_m2 = tf.square(diff[:, 1:2])
        diff_m2_dx = 0.0 # faster convergence without dx loss (covered via physics loss)
        ic_loss = tf.reduce_mean(diff_m1+diff_m2+diff_m1_dx+diff_m2_dx)

        return ic_loss
    
    def calc_physics_loss(self, x_col):
        m1_loss, m2_loss, _, _ = self.f_model(x_col)
        m1_loss_mean = tf.reduce_mean(tf.square(m1_loss))
        m2_loss_mean = tf.reduce_mean(tf.square(m2_loss))
        return [m1_loss_mean, m2_loss_mean]


    @tf.function
    def train_step(self, x_col):
        with tf.GradientTape(persistent=True) as tape:
            # data loss / initial condition
            data_loss = self.calc_loss_ic()

            # physics loss
            m1_p_loss, m2_p_loss = self.calc_physics_loss(x_col)
            combined_weighted_loss = data_loss + (self.physics_scale * (m1_p_loss + m2_p_loss))

        # retrieve gradients
        grads = tape.gradient(combined_weighted_loss, self.model.weights)
        del tape
        self.optimizer.apply_gradients(zip(grads, self.model.weights))

        m1_loss, m2_loss, pred_y_m1, pred_y_m2 = self.f_model(self.input_all)
        m1_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_y_m1) - tf.squeeze(self.y_lbl_all[:, 0])))
        m2_data_loss = tf.reduce_mean(tf.square(tf.squeeze(pred_y_m2) - tf.squeeze(self.y_lbl_all[:, 1])))
        f_pred_m1 = tf.reduce_mean(tf.square(m1_loss))
        f_pred_m2 = tf.reduce_mean(tf.square(m2_loss))
        log_data = [m1_data_loss, m2_data_loss, f_pred_m1, f_pred_m2]

        return combined_weighted_loss, log_data, [m1_loss, m2_loss, pred_y_m1, pred_y_m2]

    def fit(self, training_epochs):
        self.logger.log_train_start(self)
        for epoch in range(training_epochs):

            loss_value, log_data, pred_parameters = self.train_step(self.x_physics)

            self.logger.log_train_epoch(epoch, loss_value, log_data)
            

            if epoch % 25_000 == 0:
                self.store_intermediate_result(epoch, pred_parameters)
        self.logger.log_train_end(training_epochs, log_data)

    def predict(self, x):
        y = self.model(x)
        f_m1, f_m2, _, _ = self.f_model(x)
        return [y, f_m1, f_m2]


def get_layer_list(nr_inputs, nr_outputs, nr_hidden_layers, width):
    layers = [nr_inputs]
    for i in range(nr_hidden_layers + 1):
        layers.append(width)
    layers.append(nr_outputs)
    return layers


def setup_experiment_folders(hidden_layers, width, af_str, task_id):
    # Setting up result folder
    result_folder_name = 'res'
    os.makedirs(result_folder_name, exist_ok=True)

    # Construct experiment name
    experiment_name = "two_mass_va_"
    experiment_name += f"hl_{hidden_layers}_"
    experiment_name += f"w_{width}_"
    experiment_name += f"af_{af_str}_"
    experiment_name += f"id_{task_id}"
    print("Config name:", experiment_name)

    experiment_path = os.path.join(result_folder_name, experiment_name)
    plots_path = os.path.join(experiment_path, "plots/")
    os.makedirs(experiment_path, exist_ok=True)
    os.makedirs(plots_path, exist_ok=True)

    return experiment_path


def main():
    task_id = 5#int(sys.argv[1])
    print("task_id: ", task_id)

    # Parameters that change based on task id ############################################################################
    if task_id <= 4:
        act_func_str = "tanh"
    elif 4 < task_id <= 9:
        act_func_str = "sine"
    elif 9 < task_id <= 14:
        act_func_str = "single-sine"

    ic_points_idx =  [0]
    lr = tf.Variable(1e-4)
    hidden_layers = 7
    physics_scale = tf.Variable(1e-6)
    ######################################################################################################################
    # Fixed parameters PINN
    training_epochs = 1_200_001
    width = 128
    layers = get_layer_list(nr_inputs=1, nr_outputs=2, nr_hidden_layers=hidden_layers, width=width)

    # save_loss_freq = how often test error is saved
    logger = Logger(save_loss_freq=1, print_freq=1_000)

    # Data simulation
    c2_in = (0.5e8 * 2)
    d2_in = (1.5e3 * 2)
    m2 = 3_000
    m1 = 15_000 * 2
    x_d = [0, 1.5]  # time domain of simulation
    exp_len = 7_500  # number of points are used in training / testing the PINN
    steps = 7_500  # number of steps in time domain

    start_vec = [1.0, 0.0, 0.5, 0.0]  # pos m2, dx pos m2, pos m1, dx pos m1
    simul_results, simul_constants = get_simulated_data_two_mass(start_vec, end_time=x_d[1], steps=steps,
                                                                 exp_len=exp_len, m1=m1, m2=m2, css=c2_in, dss=d2_in,
                                                                 debug_data=True)
    y_m2_simul, y_m2_dx_simul, y_m2_dx2_simul, y_m1_simul, y_m1_dx_simul, y_m1_dx2_simul, u_simul, up_simul, tExci = simul_results
    m2, m1, c1, d1, c2, d2 = simul_constants
    #####################################

    # Getting the data
    p_start_step = 1 
    p_sampling_rate = 1 # callocation points are sampled every p_sampling_rate
    data_start = 0
    t = tExci[data_start:]
    domain = [t[0], t[-1]]

    # Initial condition data (used as data loss)
    x_initial_condition = tExci[ic_points_idx]
    y_initial_condition = np.hstack((
                        y_m1_simul[ic_points_idx], 
                        y_m2_simul[ic_points_idx],
                        y_m1_dx_simul[ic_points_idx],
                        y_m2_dx_simul[ic_points_idx])) # pos m1 (lower mass), pos m2, dx pos m1, dx pos m2

    y_label_initial_conditions = tf.constant(y_initial_condition, dtype=tf.float32)
    x_initial_condition_tensor = tf.constant(x_initial_condition, dtype=tf.float32)

    # Physics loss data
    input_data_physics = tf.convert_to_tensor(t[p_start_step:exp_len:p_sampling_rate], dtype=tf.float32)

    # Test set over the entire domain (used for logging loss, not training)
    y_labels_all = np.hstack((y_m1_simul[data_start:],  y_m2_simul[data_start:]))
    y_labels_all_tensor = tf.convert_to_tensor(y_labels_all, dtype=tf.float32)
    x_input_all_tensor = tf.convert_to_tensor(t, dtype=tf.float32)

    # Collect all tensors into a list for PINN data
    pinn_data = [x_initial_condition_tensor, y_label_initial_conditions, input_data_physics, x_input_all_tensor, y_labels_all_tensor]

    # Setting up folder structure
    storage_path = setup_experiment_folders(hidden_layers, width, act_func_str, task_id)

    plot_solution(t, y_lbl_m2=y_m2_simul[data_start:], y_lbl_m1=y_m1_simul[data_start:], x_data=x_initial_condition, y_data_m2=y_m2_simul[ic_points_idx], y_data_m1=y_m1_simul[ic_points_idx], y_pred=None, f_path_name=os.path.join(storage_path, "exact_solution"))


    pinn = PhysicsInformedNN(layers, h_activation_function=act_func_str, logger=logger, simul_constants=simul_constants,
                             domain=domain, physics_scale=physics_scale, lr=lr, data=pinn_data,
                             simul_results=simul_results, storage_path=storage_path)
    
    pinn.fit(training_epochs)

    # plot results
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, os.path.join(storage_path, "loss"), scaled=False)
    plot_loss(logger.loss_over_epoch, pinn.physics_scale, os.path.join(storage_path, "loss_scaled"), scaled=True)
    y_pred, f_m1, f_m2 = pinn.predict(x_input_all_tensor)
    plot_solution(t, y_lbl_m2=y_m2_simul[data_start:], y_lbl_m1=y_m1_simul[data_start:], x_data=x_initial_condition, y_data_m2=y_m2_simul[ic_points_idx], y_data_m1=y_m1_simul[ic_points_idx], y_pred=y_pred, f_path_name=os.path.join(storage_path, "pred_after_training"))
    print("Finished")


if __name__ == "__main__":
    main()




