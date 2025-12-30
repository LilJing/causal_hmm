from torch.utils.data import Dataset
import csv
import cv2
from torchvision import transforms, datasets
import numpy as np
import json
from PIL import Image
import torch


class RETINA(Dataset):
    def __init__(self, args, image_path, label_path, train = False):
        self.image_path = image_path
        self.label_path = label_path
        self.args = args
        self.train = train
        self.load()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.get_image_sequence(idx)
        y = torch.tensor(self.labels[idx])
        a_att = torch.tensor(self.a_atts[idx])
        b_att = torch.tensor(self.b_atts[idx])
        return image, y, a_att, b_att

    def load(self,):
        self.image_names, self.a_atts, self.b_atts, self.labels = [], [], [], []

        grade1_a_indexs = [17, 20, 21, 22, 23, 24, 25, 27, 28, 35, 36, 37, 38, 39, 40]
        grade1_b_indexs = [10, 11, 12, 13, 14, 15, 16, 18, 19, 26, 29, 30, 31, 32, 33, 34]

        interval = 32
        all_grade_num = 5

        all_grade_a_indexs, all_grade_b_indexs = [], []
        for k in range(all_grade_num):
            all_grade_a_indexs.append(np.array(grade1_a_indexs) + interval * k)
            all_grade_b_indexs.append(np.array(grade1_b_indexs) + interval * k)

        with open(self.label_path, 'r') as f:
            rdr = csv.reader(f)

            for index, row in enumerate(rdr):
                if row[0] == '':
                    break
                image_sequence_name = [row[k] for k in range(self.args.from_grade, self.args.to_grade)]
                a_atts = [[json.loads(row[all_grade_a_indexs[k][i]]) for i in range(len(grade1_a_indexs))] for k
                               in range(self.args.from_grade, self.args.to_grade)]
                self.a_atts.append(a_atts)

                b_atts = [[json.loads(row[all_grade_b_indexs[k][i]]) for i in range(len(grade1_b_indexs))] for k in
                               range(self.args.from_grade, self.args.to_grade)]

                self.b_atts.append(b_atts)

                np_label = np.array(json.loads(row[9]))

                self.image_names.append(image_sequence_name)
                self.labels.append(np_label)


    def get_image_sequence(self, idx):
        im_size = self.args.image_size
        grade_sequences = []
        for i in range(self.args.to_grade - self.args.from_grade,):
            image_path = self.image_path + '/' + str(self.image_names[idx][i])
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)
            b, g, r = cv2.split(img)
            x = cv2.merge([r, g, b])
            x = np.array(Image.fromarray(x).resize([im_size, im_size]))
            norm_trans = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

            x = Image.fromarray(np.uint8(np.array(x)))
            x = norm_trans(x)
            grade_sequences.append(x)

        return torch.stack(grade_sequences)




