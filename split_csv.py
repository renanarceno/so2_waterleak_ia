originalFilename = "hidro.csv"

test_file = "hidro_test.csv"
train_file = "hidro_train.csv"
predict_file = "hidro_pred.csv"

with open(originalFilename, "r") as file:
    i = 0
    header = next(file)

    ftest = open(test_file, "w")
    ftrain = open(train_file, "w")
    fpred = open(predict_file, "w")

    ftest.write(header)
    ftrain.write(header)
    fpred.write(header)

    for line in file:
        if i % 500 == 0:
            fpred.write(line)
        else:
            if i % 2 == 0:
                ftest.write(line)
            else:
                ftrain.write(line)
        i = i + 1
    file.close()

ftest.flush()
ftest.close()

ftrain.flush()
ftrain.close()

fpred.flush()
fpred.close()
