// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract StudentRegistry {

    // Structure
    struct Student {
        uint id;
        string name;
        uint age;
        string dept;
    }

    // Dynamic Array
    Student[] public students;

    // Fallback function
    fallback() external payable {}

    // Receive function
    receive() external payable {}

    // Add student
    function addStudent(uint id, string memory name, uint age, string memory dept) public {
        students.push(Student(id, name, age, dept));
    }

    // Get student details
    function getStudent(uint index) public view returns (uint, string memory, uint, string memory) {
        Student memory s = students[index];
        return (s.id, s.name, s.age, s.dept);
    }

    // Count students
    function count() public view returns (uint) {
        return students.length;
    }
}
