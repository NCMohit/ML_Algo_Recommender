def preprocess(file):
    file = open(file,"r")
    data = []
    for line in file.readlines():
        data.append(line.split(","))
    nomial_column_values = {}
    for i in range(len(data[0])):
        nomial_column_values[i] = []
    data = data[1:]
    for row in range(len(data)):
        for value in range(len(data[row])):
            data[row][value] = data[row][value].strip()
            if(any(l.isalpha() for l in data[row][value])):
                column = data[row].index(data[row][value])
                if(data[row][value] not in nomial_column_values[column]):
                    nomial_column_values[column].append(data[row][value])
                data[row][value] = nomial_column_values[column].index(data[row][value])
            else:
                data[row][value] = float(data[row][value])
    print("Nominal data detected: ")
    for key,value in nomial_column_values.items():
        print(key,": ",value)
    return data