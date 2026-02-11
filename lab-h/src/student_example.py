from my_classes import *


def main():
    # Create instance of StudentMarksEEEN11202
    student1 = StudentMarksEEEN11202("Alex", "12345")

    # Set assignment marks for student1
    student1.set_assignment_mark("a", 2)
    student1.set_assignment_mark("b", 5)
    # student1.set_assignment_mark("c", "apple")  # uncomment to test invalid input
    student1.set_assignment_mark("d", 2.2)
    student1.set_assignment_mark("e", 0)
    student1.set_assignment_mark("f", -3)

    # Set exam marks
    student1.set_exam_mark(65)
    student1.set_exam_mark(75.7)
    student1.set_exam_mark(101)
    student1.set_exam_mark(-6.5)

    # Print contents
    for student in [student1]:
        print(
            f"Student Name: {student.name}\n"
            f"ID: {student.id}\n"
            f"Assignment marks: {student.assignment_marks}\n"
            f"Exam mark: {student.exam_mark}\n"
            f"Overall mark: {student.overall_mark}\n"
        )


if __name__ == "__main__":
    main()
