########## OLD ONEFEATURE_METHOD1 STUFF ##########
# predictions = get_all_split_tree_predictions(model1, X1_test)
# y_pred_sum = sum_split_tree_predictions(model1, predictions)


# model1_t1 = model1.predict(X1_test, iteration_range=(0, 1), output_margin=False)
# model1_t2 = model1.predict(X1_test, iteration_range=(1, 2), output_margin=False)
# model1_t3 = model1.predict(X1_test, iteration_range=(2, 3), output_margin=False)
# y_tree1 = np.array(model1_t1)
# print(y_tree1)
# y_tree2 = np.array(model1_t2)
# y_tree3 = np.array(model1_t3)
# y1_sum = y_tree1 + y_tree2 + y_tree3 - 2 * bias
# print(y1_sum[:10])
# print(y1_3[:10])
# bias = get_bias(model1)

# EACH INDIVIDUAL TREE (INCLUDES BIAS EACH TIME):
# plt.scatter(x1, y_tree1, label="tree 1 predicted y", color="blue", marker=".")
# plt.scatter(x1, y_tree2, label="tree 2 predicted y", color="brown", marker=".")
# plt.scatter(x1, y_tree3, label="tree 3 predicted y", color="purple", marker=".")
