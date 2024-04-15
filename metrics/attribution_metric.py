attrs = {'color': {'silver', 'blue', 'dark', 'orange', 'teal', 'pink', 'cream colored', 'evergreen', 'maroon', 'navy',
                   'brass', 'dark colored', 'translucent', 'black', 'light blue', 'purple', 'gray', 'blond', 'yellow',
                   'light colored', 'black and white', 'white', 'green', 'bronze', 'beige', 'red', 'tan', 'ivory',
                   'light brown', 'chocolate', 'dark brown', 'dark blue', 'brown', 'transparent', 'khaki',
                   'rainbow colored', 'brunette', 'gold'},
         'material': {'wool', 'cobblestone', 'denim', 'brick', 'asphalt', 'styrofoam', 'cotton', 'paper', 'porcelain',
                      'stainless steel', 'marble', 'granite', 'steel', 'wood', 'silk', 'tin', 'chrome', 'brass', 'clay',
                      'rock', 'metal', 'straw', 'wire', 'copper', 'cardboard', 'mesh', 'bamboo', 'rubber', 'aluminum',
                      'bronze', 'leather', 'wicker', 'glass', 'hardwood', 'stone', 'cloth', 'concrete', 'iron',
                      'plastic', 'lace', 'crystal'},
         "shape": {"round", "square", "long", "rectangular", "triangular", "short", "folded", "rounded", "recessed",
                   "curled", "octagonal", "tall", "thin", "thick", "skinny", "fat"},
         "activity": {'sleeping', 'waiting', 'getting', 'staring', 'dragging', 'carrying', 'looking up', 'observing',
                      'painting', 'throwing', 'helping', 'drawing', 'jumping', 'traveling', 'reflecting', 'lying',
                      'grazing', 'looking', 'pulling', 'herding', 'covering', 'hitting', 'talking', 'tying', 'cutting',
                      'watching', 'following', 'petting', 'having meeting', 'eating', 'playing',
                      'posing', 'making', 'licking', 'using', 'reaching', 'touching', 'balancing',
                      'biting', 'shopping', 'feeding', 'drinking', 'climbing', 'preparing',
                      'dining', 'pushing', 'driving', 'wedding', 'coming', 'walking', 'swinging',
                      'leaving', 'sitting', 'taking', 'steering', 'approaching', 'working', 'baking',
                      'cleaning', 'picking', 'hanging', 'frosting', 'facing', 'catching',
                      'vending', 'dressing', 'wearing', 'reading', 'tossing', 'serving', 'brushing teeth', 'pointing',
                      'wading', 'chasing', 'guiding', 'looking down', 'flying', 'standing', 'parking', 'chewing',
                      'growing', 'going', 'something', 'grabbing', 'selling', 'typing', 'smiling', 'cooking', 'exiting',
                      'bending', 'resting', 'crossing', 'sinking', 'splashing', 'boarding', 'holding'},
         "size": {"tiny", "large", "small", "huge", "little", "giant"},
         "pose": {"lying", "standing", "pointing", "crouching", "jumping", "sitting", "bending", 'leaning', 'folding',
                  'kneeling', 'floating', 'crouching'},
         "sport": {"snowboarding", "skateboarding", "riding", "skiing", "surfing", "skating", "swimming", "performing",
                   'running', 'baseball', 'tennis', 'soccer', 'basketball', 'football'},
         "direction": {"top", "bottom", "left", "right", "behind", "front", "under", "above", "below"},
         "animal": {"cat", "cats", "dog", "dogs", "horse", "horses", "cows", "cow", "goose", "geese", "bird", "birds",
                    "bear", "bears", "sheep", "elephant", "elephants", "giraffe", "giraffes", "donkey",
                    "donkeys", "zebra", "zebras", "eagle", "eagles", "goat", "goats", "duck", "ducks", "deer", "deer",
                    "lion", "lions", "frisbee", "frisbees", "giraffe", "lamb", "lambs", "kitten", "kittens", "dragon",
                    "dragons", "calf", "calves", "crab", "crabs", "alligator", "alligators", "gorilla", "gorillas",
                    "butterfly", "butterflies", "bull", "bulls", "puppy", "puppies", "chicken", "chickens", "swan",
                    "swans", "rhino", "rhinos", "snail", "snails", "frog", "frogs", "monkey", "monkeys", "ostrich",
                    "ostriches", "owl", "owls", "kine", "seagull", "seagulls", "pigeon", "pigeons", "antelope",
                    "antelopes", "parrot", "parrots"},
         "person": {"person", "people", "boy", "boys", "girl", "girls", "man", "men", "woman", "women", "skateboarder",
                    "skateboarders", "player", "players", "skater", "skaters", "batter", "batters", "pitcher",
                    "pitchers", "catcher", "catchers", "him", "her", "skier", "skiers", "driver", "drivers", "speaker",
                    "speakers", "lady", "ladies", "gentleman", "gentlemen", "guy", "guys", "snowboarder", "snowboarders"
                    },
         "plant": {"grass", "flower", "flowers", "onion", "onions", "tomato", "tomatoes", "potato", "potatoes",
                   "carrot", "carrots", "cucumber", "cucumbers", "avocado", "avocados", "bean", "beans", "broccoli",
                   "lemon", "lemons", "spinach", "corn", "corns", "pear", "pears", "apple", "apples", "pepper",
                   "peppers", "rice", "pickle", "pickles", "cauliflower", "lettuce", "basil", "garlic", "garlics",
                   "banana", "bananas", "parsley", "cabbage", "zucchini", "celery"}
         }


class attr_evaluater():
    def __init__(self, gts):
        self.attr_dic = {attr: {} for attr in attrs}
        for qid, gt in gts.items():
            exp = ' ' + gt['explanation'].replace('.', '').replace(',', '').replace(';', '').lower() + ' '
            for attr, v in attrs.items():
                for word in v:
                    if ' ' + word + ' ' in exp and qid not in self.attr_dic[attr]:
                        self.attr_dic[attr][qid] = [word]
                    elif ' ' + word + ' ' in exp:
                        self.attr_dic[attr][qid].append(word)

    def score(self, res):
        res = {k: ' ' + v['explanation'].replace('.', '').replace(',', '').replace(';', '').lower() + ' ' for k, v in res.items()}
        score = {attr: [] for attr in attrs}
        for attr in attrs:
            for qid, words in self.attr_dic[attr].items():
                for word in words:
                    if ' ' + word + ' ' in res[qid]:
                        score[attr].append(1)
                    else:
                        score[attr].append(0)
            score[attr] = sum(score[attr]) / len(score[attr]) * 100 if len(score[attr]) > 0 else 0

        return score
