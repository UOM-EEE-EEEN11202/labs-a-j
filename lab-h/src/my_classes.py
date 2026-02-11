import numpy as np


class NoSubmission:
    """
    Class to represent no submission made.
    """

    def __init__(self):
        self.value = np.nan

    def __str__(self):
        return "No Submission"

    def __repr__(self):
        return "No Submission"


class StudentMarksEEEN11202:
    """
    Docstring for StudentMarksEEEN11202
    """

    def __init__(self, name, id):
        no_assignments = 20
        self.name = name
        self.id = id
        self.assignment_marks = np.full((no_assignments,), NoSubmission())
        self.exam_mark = NoSubmission()
        self.overall_mark = NoSubmission()

    def set_exam_mark(self, mark):
        self.exam_mark = mark
        self._set_overall_mark()

    def set_assignment_mark(self, assignment_letter, mark):
        index = ord(assignment_letter.lower()) - ord("a")  # adjust for 0-based indexing
        self.assignment_marks[index] = mark
        self._set_overall_mark()

    def _set_overall_mark(self):
        assignment_weight = 0.5
        exam_weight = 0.5

        # Set exam to zero if no submission
        tmp_exam_mark = self.exam_mark
        if isinstance(tmp_exam_mark, NoSubmission):
            tmp_exam_mark = 0

        # Set assignments to zero if no submission, and sum
        tmp_assignment_mark = 0
        for assignment_mark in self.assignment_marks:
            if isinstance(assignment_mark, NoSubmission):
                tmp_assignment_mark += 0
            else:
                tmp_assignment_mark += assignment_mark

        # Calculate overall mark
        self.overall_mark = (
            assignment_weight * tmp_assignment_mark + exam_weight * tmp_exam_mark
        )

        return self.overall_mark
