import os, argparse

from omegaconf import OmegaConf


class Synset:
    next_id = 0

    def __init__(self, wnid, gloss="", words=None, children=None, parents=None):
        self.wnid = wnid
        self.gloss = gloss
        self.words = words if words is not None else []
        self.children = children if children is not None else []
        self.parents = parents if parents is not None else []
        self.marked = False
        self.in_imagenet = False
        self.count_train = None
        self.count_val = None
        self.depth = float("inf")
        self.sub_classes = []
        self.id = None

    def calc_depth(self, depth):
        if depth < self.depth:
            self.depth = depth
            for child in self.children:
                child.calc_depth(depth+1)

    def replace_ptr(self, synsets):
        for k, child in enumerate(self.children):
            self.children[k] = synsets[child]

        for k, parent in enumerate(self.parents):
            self.parents[k] = synsets[parent]

    def mark(self):
        if not self.marked:
            self.marked = True
            for parent in self.parents:
                parent.mark()

    def prune_unmarked(self):
        new_children = []
        for child in self.children:
            if child.marked:
                new_children.append(child)
        self.children = new_children

    def prune_parents(self):
        if len(self.parents) > 1:
            best_parent = self.parents[0]
            for parent in self.parents[1:]:
                if len(parent.children) < len(best_parent.children):
                    best_parent = parent
            self.parents = [best_parent]

        new_children = []
        for child in self.children:
            child.prune_parents()
            if self in child.parents:
                new_children.append(child)
        self.children = new_children

    def merge(self):
        if self.is_leaf:
            return

        self.count_train = 0 if self.count_train is None else self.count_train
        self.count_val = 0 if self.count_val is None else self.count_val
        for child in self.children:
            child.merge()
            self.count_train += child.count_train
            self.count_val += child.count_val
            if child.in_imagenet:
                self.sub_classes.append(child.wnid)
            if len(child.sub_classes) > 0:
                self.sub_classes.extend(child.sub_classes)

        self.children = []

    def set_id(self):
        if self.is_leaf:
            self.id = Synset.next_id
            Synset.next_id += 1
        
        for child in self.children:
            child.set_id()


    def merge_by_count(self, min_count):
        if self.is_leaf:
            return

        for child in self.children:
            child.merge_by_count(min_count)
            if child.is_leaf and child.count_train < min_count:
                self.merge()
                return

    def merge_by_depth(self, max_depth):
        if self.depth >= max_depth:
            self.merge()
        else:
            for child in self.children:
                child.merge_by_depth(max_depth)

    def merge_bottom_up(self, num_merges):
        if self.is_leaf:
            return num_merges

        rem_merges = num_merges
        for child in self.children:
            ret = child.merge_bottom_up(num_merges)
            rem_merges = min(rem_merges, ret)

        if rem_merges > 0:
            self.merge()

        return max(rem_merges - 1, 0)
    
    def merge_unbranching(self):
        if self.is_leaf:
            return True

        if len(self.children) == 1:
            if self.children[0].merge_unbranching():
                self.merge()
                return True
        else:
            for child in self.children:
                child.merge_unbranching()
            return False


    def _str_child(self, depth, max_depth=4):
        pad = "\t"*depth
        res = f"{pad}{self._str_parent()}"
        if depth < max_depth:
            for child in self.children:
                if type(child) == str:
                    res += "\n" + pad + "\t" + child
                else:
                    res += "\n" + child._str_child(depth+1)
        return res
    
    def _str_parent(self):
        return f"{self.wnid}  ({', '.join(self.words)})"

    def __str__(self):
        res = f"{self.parents}"
        res += " -> "
        res += self._str_child(0)
        return res

    def __repr__(self):
        return f"Synset({repr(self.wnid)})"

    def to_dict(self):
        res = {
            "wnid": self.wnid,
            "words": self.words,
            "gloss": self.gloss,
            "depth": self.depth,
        }

        if not self.is_leaf:
            res["children"] = [ch.to_dict() for ch in self.children]
        else:
            if len(self.sub_classes) > 0:
                res["sub_classes"] = self.sub_classes
            res["train_examples"] = self.count_train
            res["val_examples"] = self.count_val
            res["id"] = self.id

        return res
    
    def to_list(self):
        res = [self]
        for child in self.children:
            res.extend(child.to_list())
        return res

    def inverse(self):
        assert self.is_leaf
        stats = {
            "words": self.words,
            "id": self.id,
            "wnid": self.wnid,
        }
        if len(self.sub_classes) == 0:
            return {self.wnid: stats}
        else:
            return {wnid: stats for wnid in self.sub_classes}
            

    @property
    def is_leaf(self):
        return len(self.children) == 0

def parse_line(line):
    line, gloss = line.split("|")
    line = line.split(" ")

    res = dict()

    res["wnid"] = "n" + line[0]
    res["gloss"] = gloss.strip()

    num_words = int(line[3], base=16)
    words = []
    k = 4
    for _ in range(num_words):
        words.append(line[k].replace("_", " "))
        k += 2
    res["words"] = words
    
    num_ptr = int(line[k])
    parents = []
    children = []
    k += 1
    for _ in range(num_ptr):
        symb = line[k]
        other_id = "n" + line[k+1]
        pos = line[k+2]
        source = line[k+3]
        k += 4
        if source != "0000": continue
        if pos != "n": continue
        if symb[0] == "@":
            parents.append(other_id)
        if symb[0] == "~":
            children.append(other_id)
    res["children"] = children
    res["parents"] = parents

    return Synset(**res)

def parse_wordnet(filename):
    with open(filename) as file:
        lines = file.read().rstrip().split("\n")

    synsets = dict()
    for k, line in enumerate(lines):

        #skip license
        if line[0] == " ":
            continue 

        synset = parse_line(line)
        synsets[synset.wnid] = synset

    for wnid in synsets:
        synsets[wnid].replace_ptr(synsets)

    return synsets

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--plot",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="Plot number of training/validation examples per node and depths",
    )
    parser.add_argument(
        "--imagenet_classes",
        type=str,
        default="imagenet_counts.txt",
        help="Path to file with either wordnet-ids or (wordnet-ids + number of images)",
        nargs="?",
    )
    parser.add_argument(
        "--wordnet_nouns",
        type=str,
        default="data.noun",
        help="Path to noun wordnet database",
        nargs="?",
    )
    parser.add_argument(
        "--merge_method",
        type=str,
        default="bottom-up",
        help="How to merge? bottom-up (1), depth (2) or count (3)",
        nargs="?",
    )
    parser.add_argument(
        "--merge_parameter",
        type=str,
        default=None,
        help="Parameter for merging",
        nargs="?",
    )
    opt = parser.parse_args()


    db = parse_wordnet(opt.wordnet_nouns)

    roots = [db[wnid] for wnid in db if len(db[wnid].parents) == 0]
    for root in roots:
        root.calc_depth(0)
    root.prune_parents()

    if os.path.exists(opt.imagenet_classes):
        with open(opt.imagenet_classes) as file:
            imagenet = file.read().rstrip().split("\n")
    else:
        with open("imagenet_synsets.txt") as file:
            imagenet = file.read().rstrip().split("\n")

    for line in imagenet:
        if " " in line:
            wnid, count_train, count_val = line.split(" ")
        else:
            wnid = line
            count_train, count_val = 0, 0

        db[wnid].in_imagenet = True
        db[wnid].mark()
        db[wnid].count_train = int(count_train)
        db[wnid].count_val   = int(count_val)

    print("Pre-merge stats:")
    print(f"{len(db)} synsets in wordnet")
    marked_db = dict()
    for wnid in db:
        if db[wnid].marked:
            marked_db[wnid] = db[wnid]
    db = marked_db

    for wnid in db:
        db[wnid].prune_unmarked()

    print(f"{len(db)} synsets in (ImageNet + Metasynsets)")

    roots = [db[wnid] for wnid in db if len(db[wnid].parents) == 0]
    for root in roots:
        root.calc_depth(0)
        root.prune_parents()
    print("Max Depth:", max(db[wnid].depth for wnid in db))

    print("Leafs:", sum(db[wnid].is_leaf for wnid in db))
    for root in roots:
        root.merge_unbranching()
        if opt.merge_method.lower() == "bottom-up" or opt.merge_method == "1":
            root.merge_bottom_up(opt.merge_parameter or 6)
        elif opt.merge_method.lower() == "depth"   or opt.merge_method == "2":
            root.merge_by_depth(opt.merge_parameter or 6)
        elif opt.merge_method.lower() == "count"   or opt.merge_method == "3":
            root.merge_by_count(opt.merge_parameter or 1500)
        root.merge_unbranching()
        root.set_id()
    
    db = {x.wnid: x for x in root.to_list()}
    leafs = [db[wnid] for wnid in db if db[wnid].is_leaf]

    leafs.sort(key=lambda leaf: leaf.count_train, reverse=True)
    leaf_info = ""
    for leaf in leafs:
        leaf_info += ", ".join(leaf.words) + "\n"
        leaf_info += leaf.gloss + "\n"
        leaf_info += str(leaf.count_train) + ", " + str(leaf.count_val) + "\n\n"

    with open("leaf_info.txt", "w") as file:
        file.write(leaf_info)

    print("\nAfter-merge stats:")
    print(f"{len(db)} synsets after merging")
    print("Max Depth:", max(db[wnid].depth for wnid in db))
    print("Leafs:", len(leafs))
    print("Unmerged leafs:", sum(1 for leaf in leafs if len(leaf.sub_classes) <= 1))

    json = [root.to_dict() for root in roots]
    conf = OmegaConf.create(json)
    yaml = OmegaConf.to_yaml(conf)

    with open("out.yaml", "w") as file:
        file.write(yaml)

    inverse = {}
    for leaf in leafs:
        inverse.update(leaf.inverse())

    yaml = OmegaConf.to_yaml(OmegaConf.create(inverse))
    with open("class_dict.yaml", "w") as file:
        file.write(yaml)

    if opt.plot:
        from matplotlib import pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(1, 3)
        ax[0].set_title("#train images")
        sns.stripplot(y = [leaf.count_train for leaf in leafs], ax=ax[0])
        ax[1].set_title("#validation images")
        sns.stripplot(y = [leaf.count_val for leaf in leafs], ax=ax[1])
        ax[2].set_title("leaf depth")
        sns.stripplot(y = [leaf.depth for leaf in leafs], ax=ax[2])
        plt.tight_layout()
        plt.show()
