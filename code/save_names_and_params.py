from argparse import ArgumentParser
import pandas as pd
import numpy as np
import utils


def load_faculty_names():
    syllabus_path = utils.get_syllabus_path()

    syllabus = pd.read_excel(syllabus_path)
    faculty_names = syllabus["Lecturer"].values
    faculty_names = [n for n in faculty_names if (type(n) is str) and (n != "TAs")]
    faculty_names = list(np.unique(faculty_names))

    faculty_names.pop(1)
    faculty_names.append("Andre Fenton")

    return faculty_names


def load_TA_names():
    return ["Jeff Johnston", "Erfan Zabeh", "Chiara Mastrogiuseppe", "Shuqi Wang", "Yue Kris Wu", "Alan Lai"]


def load_student_names():
    names = [
        "Beau, Maxime",
        "Bigus, Erin",
        "Blair, Garrett",
        "Buck, Friederike",
        "Chen, Kevin",
        "Elnozahy, Sarah",
        "Grier, Harrison",
        "Jing, Zeyu",
        "Kajikawa, Koichiro",
        "Landau, Andrew",
        "Majumder, Shouvik",
        "Maoutsa, Dimitra",
        "Mirbagheri, Saghar",
        "Natrajan, Maanasa",
        "Nie, Chechang",
        "Osman, Mo",
        "Park, JeongJun",
        "Sakelaris, Bennet",
        "Shi, Yuelin",
        "Singer-Clark, Tyler",
        "Teasley, Audrey",
        "Thurston, Mackenzie",
        "Tian, Gengshuo",
        "Tiberi, Lorenzo",
        "Wilkins, Lillian",
        "Yang, Zidan",
    ]
    return [" ".join(n.split(", ")[::-1]) for n in names]


if __name__ == "__main__":
    # Handle user options
    parser = ArgumentParser(description="Save names and parameters for the Hopfield network")
    parser.add_argument("--full-name", default=False, action="store_true", help="Use full names instead of first names")
    parser.add_argument("--min-saturation", type=int, default=200, help="Minimum saturation for the colors")
    parser.add_argument("--horizontal-padding", type=int, default=18, help="Horizontal padding for random positioning")
    parser.add_argument("--vertical-padding", type=int, default=4, help="Vertical padding for random positioning")
    args = parser.parse_args()

    # Get names as a list
    faculty = load_faculty_names()
    TAs = load_TA_names()
    students = load_student_names()

    # Assign roles to each name
    faculty_role = ["Faculty"] * len(faculty)
    TA_role = ["TA"] * len(TAs)
    student_role = ["Student"] * len(students)

    # Create master lists
    names = faculty + TAs + students
    first_names = [n.split(" ")[0] for n in names]
    last_names = [n.split(" ")[-1] for n in names]
    roles = faculty_role + TA_role + student_role

    # Assign colors to each name
    r, g, b = [], [], []
    for i in range(len(names)):
        h = i / len(names)
        cr, cg, cb = utils.generate_saturated_color(min_saturation=args.min_saturation)
        r.append(cr)
        g.append(cg)
        b.append(cb)

    # Get positions for each name
    left, top, right, bottom = utils.get_offsets(names if args.full_name else first_names)
    width = right - left  # np.minimum(0, left)
    height = bottom - top  # np.minimum(0, top)
    max_width = width.max()
    max_height = height.max()

    # Calculate the image size based on the largest name and the desired padding
    image_width = max_width + 2 * args.horizontal_padding
    image_height = max_height + 2 * args.vertical_padding

    max_x_padding = args.horizontal_padding + (max_width - width) // 2
    max_y_padding = args.vertical_padding + (max_height - height) // 2

    # Pick random offsets for each name
    x_offsets = np.random.randint(-max_x_padding, max_x_padding - 2, len(names))
    y_offsets = np.random.randint(-max_y_padding, max_y_padding - 2, len(names))

    # Save the data
    data = dict(
        first_names=first_names,
        last_names=last_names,
        roles=roles,
        r=r,
        g=g,
        b=b,
        x_offsets=x_offsets,
        y_offsets=y_offsets,
        image_width=image_width,
        image_height=image_height,
        args=vars(args),
    )
    np.save(utils.get_name_data_path(), data)
