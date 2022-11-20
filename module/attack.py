import numpy as np


def attackModel(
    dataset, test_x, test_y, model, attack_model, sample_size=5000, test_size=200
):
    """
    It takes a dataset, a model, and an attack model, and returns a list of indices of the test set, the
    original test set, the adversarial test set, and the distance between the original and adversarial
    test set
    
    `dataset:` dataset object
    `test_x:` test data
    `test_y:` labels of the test set
    `model:` model to be attacked
    `attack_model:` attack model
    `sample_size:` number of samples to attack, defaults to 5000 (optional)
    `test_size:` number of samples to attack, defaults to 200 (optional)
    
    `return` test_list,orig_list,adv_list,dist_list
    """
    SAMPLE_SIZE = sample_size
    TEST_SIZE = test_size
    test_idx = np.random.choice(len(dataset.test_y), SAMPLE_SIZE, replace=False)

    test_list = []
    orig_list = []
    orig_label_list = []
    adv_list = []
    dist_list = []
    for i in range(SAMPLE_SIZE):
        x_orig = test_x[test_idx[i]]
        orig_label = test_y[test_idx[i]]

        orig_preds = model.predict(x_orig)[0]
        # print(orig_label, orig_preds, np.argmax(orig_preds))
        if np.argmax(orig_preds) != orig_label:
            # print('skipping wrong classifed ..')
            # print('--------------------------')
            continue
        x_len = np.sum(np.sign(x_orig))
        if x_len >= 100:
            # print('skipping too long input..')
            # print('--------------------------')
            continue
        # if np.max(orig_preds) < 0.90:
        #    print('skipping low confidence .. \n-----\n')
        #    continue
        print("****** ", len(test_list) + 1, " ********")
        orig_list.append(x_orig)
        target_label = 1 if orig_label == 0 else 0
        orig_label_list.append(orig_label)
        x_adv = attack_model.attack(x_orig, target_label)
        adv_list.append(x_adv)
        if x_adv is None:
            print("%d failed" % (i + 1))
            dist_list.append(100000)
        else:
            num_changes = np.sum(x_orig != x_adv)
            print("%d - %d changed." % (i + 1, num_changes))
            dist_list.append(num_changes)
            # display_utils.visualize_attack(sess, model, dataset, x_orig, x_adv)
        print("--------------------------")
        if len(orig_list) >= TEST_SIZE:
            break
    return orig_list,adv_list,dist_list
