#[derive(Debug, Clone)]
struct Criteria {
    axis: usize,
    split: f64,
}

#[derive(Debug, Clone)]
struct Class(String);

#[derive(Debug, Clone)]
enum Tree {
    Branch(Criteria, Box<Tree>, Box<Tree>),
    Leave(Class),
}

fn loss() {}

fn gradient_descent() {}
