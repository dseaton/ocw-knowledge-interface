import gensim.parsing.preprocessing as gpp

def gensim_custom_preprocess(text):
    """
    Prepare text for topic extraction. Creating a single function so it can be used
    in multiple contexts (e.g., when we need to process user input from a dash interface).
    """
    CUSTOM_STOPWORDS = [] # Not currently utilized

    ### Gensim Preprocess
    CUSTOM_FILTERS = [
        lambda x: x.lower(), #lowercase
        gpp.strip_multiple_whitespaces,# remove repeating whitespaces
        gpp.strip_numeric, # remove numbers
        gpp.remove_stopwords,# remove stopwords
        gpp.strip_short, # remove words less than minsize=3 characters long
        gpp.strip_punctuation,
    #     gpp.stem_text # return porter-stemmed text,
    ]

    return gpp.preprocess_string(text, filters=CUSTOM_FILTERS)

find_by_examples = [
    "Deep Learning", "Computation", "Artificial Intelligence",
    "Astrophysics", "Electromagnetism", "Maxwell", 
    "Relativity", "Thermodynamics", "Gluon",
    "Statistics", "Statistical Mechanics",
    "Programming", "Electronics", "Signal Processing", "Nanotechnology", 
    "Architecture", "Music",
    "Medical Imaging", "Literature", "Fiction",
    "Linear Algebra", "Topology", "Calculus", "Probability", 
    "Biophysics", "Organic Chemistry", "Anthropology",
    "Econometrics", "Economics", "Project Management", "Innovation",
]

find_by_topic_second_level = [
    "Accounting",
    "Aerospace Engineering",
    "African-American Studies",
    "Algebra and Number Theory",
    "Anatomy and Physiology",
    "Anthropology",
    "Applied Mathematics",
    "Archaeology",
    "Architecture",
    "Art History",
    "Asian Studies",
    "Biological Engineering",
    "Biology",
    "Biomedical Enterprise",
    "Biomedical Instrumentation",
    "Biomedical Signal and Image Processing",
    "Biomedicine",
    "Business Ethics",
    "Calculus",
    "Cancer",
    "Cellular and Molecular Medicine",
    "Chemical Engineering",
    "Chemistry",
    "Civil Engineering",
    "Cognitive Science",
    "Communication",
    "Computation",
    "Computer Science",
    "Curriculum and Teaching",
    "Differential Equations",
    "Discrete Mathematics",
    "Earth Science",
    "Economics",
    "Education Policy",
    "Educational Technology",
    "Electrical Engineering",
    "Entrepreneurship",
    "Environmental Engineering",
    "Epidemiology",
    "European and Russian Studies",
    "Finance",
    "Functional Genomics",
    "Game Design",
    "Game Theory",
    "Gender Studies",
    "Geography",
    "Global Poverty",
    "Globalization",
    "Health and Exercise Science",
    "Health Care Management",
    "Higher Education",
    "History",
    "Immunology",
    "Indigenous Studies",
    "Industrial Relations and Human Resource",
    "Information Technology",
    "Innovation",
    "Language",
    "Latin and Caribbean Studies",
    "Leadership",
    "Legal Studies",
    "Linear Algebra",
    "Linguistics",
    "Literature",
    "Management",
    "Marketing",
    "Materials Science and Engineering",
    "Mathematical Analysis",
    "Mathematical Logic",
    "Mechanical Engineering",
    "Media Studies",
    "Medical Imaging",
    "Mental Health",
    "Middle Eastern Studies",
    "Music",
    "Nanotechnology",
    "Nuclear Engineering",
    "Ocean Engineering",
    "Operations Management",
    "Organizational Behavior",
    "Pathology and Pathophysiology",
    "Performance Arts",
    "Pharmacology and Toxicology",
    "Philosophy",
    "Physical Education and Recreation",
    "Physics",
    "Political Science",
    "Probability and Statistics",
    "Project Management",
    "Psychology",
    "Public Administration",
    "Public Health",
    "Real Estate",
    "Religion",
    "Sensory-Neural Systems",
    "Social Medicine",
    "Sociology",
    "Spectroscopy",
    "Speech Pathology",
    "Subtopic",
    "Supply Chain Management",
    "Systems Engineering",
    "The Developing World",
    "Topology and Geometry",
    "Urban Studies",
    "Visual Arts",
    "Women's Studies",
]
