# SQL essentials for tech coding interview


## Reference
- https://www.geeksforgeeks.org/sql-interview-questions/

## Common template
```sql
SELECT c1, c2, ... FROM t1
WHERE c2 = 1 AND c4 = 2
GROUP BY ...
HAVING count(*) > 1
ORDER BY c2
``` 

## JOIN tables
```sql
SELECT FirstName, LastName, City, State
FROM Person LEFT JOIN Address
ON Person.PersonId = Address.PersonId
```

## DISTINCT, LIMIT, OFFSET
```sql
SELECT
    (SELECT DISTINCT
            Salary
        FROM
            Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1) AS SecondHighestSalary
```

## IFNULL
```sql
SELECT
    IFNULL(
        (SELECT DISTINCT Salary
        FROM Employee
        ORDER BY Salary DESC
        LIMIT 1 OFFSET 1),
    NULL) AS SecondHighestSalary
```

## FUNCTION
```sql
CREATE FUNCTION getNthHighestSalary(N INT) RETURNS INT
BEGIN
  DECLARE M INT;
  SET M=N-1;
  RETURN (
      # Write your MySQL query statement below.
      SELECT DISTINCT Salary FROM Employee ORDER BY Salary DESC LIMIT M, 1
      # short from for "LIMIT 1 OFFSET M"
  );
END
```

## Variable
```sql
SELECT
  Score,
  @rank := @rank + (@prev <> (@prev := Score)) AS Rank
FROM
  Scores,
  (SELECT @rank := 0, @prev := -1) INIT
ORDER BY Score desc

SELECT Score, 
       (SELECT COUNT(DISTINCT Score)+1 
       FROM Scores
       WHERE Score > t1.Score) AS Rank
FROM Scores t1 
ORDER BY Score DESC
```

## WHERE (IN, NOT IN, > (SELECT ...))
```sql
SELECT
    Department.name AS 'Department',
    Employee.name AS 'Employee',
    Salary
FROM
    Employee
        JOIN
    Department ON Employee.DepartmentId = Department.Id
WHERE
    (Employee.DepartmentId , Salary) IN # SELECT clause; NOT IN
        (SELECT
            DepartmentId, MAX(Salary)
        FROM
            Employee
        GROUP BY DepartmentId
        )
;

SELECT
    d.Name AS 'Department', e1.Name AS 'Employee', e1.Salary
FROM
    Employee e1
        JOIN
    Department d ON e1.DepartmentId = d.Id
WHERE
    3 > (SELECT # SELECT clause
            COUNT(DISTINCT e2.Salary)
        FROM
            Employee e2
        WHERE
            e2.Salary > e1.Salary
                AND e1.DepartmentId = e2.DepartmentId
        )
;
```