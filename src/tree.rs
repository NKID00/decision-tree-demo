use core::f64;
use std::{
    collections::HashMap,
    ops::{Deref, Index, IndexMut},
};

const MIN_ELEMENTS_TO_SPLIT: usize = 3;
const MAX_DEPTH: usize = 7;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Axis(pub i32);

#[derive(Debug, Clone, Copy)]
pub struct Criterion {
    pub axis: Axis,
    pub split: f64,
}

impl Criterion {
    fn split(
        &self,
        data: &Vec<(DataPoint, Class)>,
    ) -> (Vec<(DataPoint, Class)>, Vec<(DataPoint, Class)>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for (dp, class) in data {
            if dp[self.axis] < self.split {
                left.push((dp.clone(), *class));
            } else {
                right.push((dp.clone(), *class));
            }
        }
        (left, right)
    }
    fn classify(&self, data: &Vec<DataPoint>) -> (Vec<DataPoint>, Vec<DataPoint>) {
        let mut left = Vec::new();
        let mut right = Vec::new();
        for dp in data {
            if dp[self.axis] < self.split {
                left.push(dp.clone());
            } else {
                right.push(dp.clone());
            }
        }
        (left, right)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Class(pub i32);

impl Deref for Class {
    type Target = i32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub struct DataPoint(pub Vec<f64>);

impl Index<Axis> for DataPoint {
    type Output = f64;

    fn index(&self, index: Axis) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl Index<&Axis> for DataPoint {
    type Output = f64;

    fn index(&self, index: &Axis) -> &Self::Output {
        &self.0[index.0 as usize]
    }
}

impl IndexMut<Axis> for DataPoint {
    fn index_mut(&mut self, index: Axis) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

impl IndexMut<&Axis> for DataPoint {
    fn index_mut(&mut self, index: &Axis) -> &mut Self::Output {
        &mut self.0[index.0 as usize]
    }
}

impl Deref for DataPoint {
    type Target = Vec<f64>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[derive(Debug, Clone)]
pub enum DecisionTree {
    Branch(Criterion, Box<DecisionTree>, Box<DecisionTree>),
    Leave(Class),
}

fn gini_loss(classes: &[Class], vec: &[(DataPoint, Class)]) -> f64 {
    let mut loss = 0f64;
    for class in classes {
        let p = 1f64 / (vec.len() as f64)
            * (vec.iter().filter(|(_dp, cls)| cls == class).count() as f64);
        loss += p * (1f64 - p);
    }
    loss
}

fn find_criterion(axises: &[Axis], classes: &[Class], data: &Vec<(DataPoint, Class)>) -> Criterion {
    let mut min_impurity = f64::MAX;
    let mut min_impurity_axis = axises[0];
    let mut min_impurity_split = 0f64;
    for axis in axises {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        for (dp, _class) in data {
            max = max.max(dp[axis]);
            min = min.min(dp[axis]);
        }
        let step = (max - min) / 100.;
        if step == 0. {
            continue;
        }
        let mut split_criterion = min + step;
        while split_criterion <= max {
            let c = Criterion {
                axis: *axis,
                split: split_criterion,
            };
            let (left, right) = c.split(data);
            if left.is_empty() || right.is_empty() {
                split_criterion += step;
                continue;
            }
            let left_loss = gini_loss(classes, &left);
            let right_loss = gini_loss(classes, &right);
            let impurity = ((left.len() as f64) / (data.len() as f64)) * left_loss
                + ((right.len() as f64) / (data.len() as f64)) * right_loss;
            if impurity < min_impurity {
                min_impurity = impurity;
                min_impurity_axis = *axis;
                min_impurity_split = split_criterion;
            }
            split_criterion += step;
        }
    }
    Criterion {
        axis: min_impurity_axis,
        split: min_impurity_split,
    }
}

#[derive(Debug, Clone)]
enum TrainTree {
    Branch(Criterion, Box<TrainTree>, Box<TrainTree>),
    Leave(Vec<(DataPoint, Class)>),
}

impl From<TrainTree> for DecisionTree {
    fn from(value: TrainTree) -> Self {
        match value {
            TrainTree::Branch(criterion, train_tree, train_tree1) => DecisionTree::Branch(
                criterion,
                Box::new(DecisionTree::from(*train_tree)),
                Box::new(DecisionTree::from(*train_tree1)),
            ),
            TrainTree::Leave(vec) => {
                let mut counter = HashMap::<Class, usize>::new();
                for (_dp, class) in vec {
                    match counter.get_mut(&class) {
                        Some(c) => *c += 1,
                        None => {
                            counter.insert(class, 1);
                        }
                    }
                }
                DecisionTree::Leave(
                    *counter
                        .iter()
                        .max_by_key(|(_class, count)| *count)
                        .unwrap()
                        .0,
                )
            }
        }
    }
}

fn grow(
    axises: &[Axis],
    classes: &[Class],
    data: Vec<(DataPoint, Class)>,
    depth: usize,
) -> TrainTree {
    if depth < MAX_DEPTH && data.len() > MIN_ELEMENTS_TO_SPLIT {
        let c = find_criterion(axises, classes, &data);
        let (left, right) = c.split(&data);
        TrainTree::Branch(
            c,
            Box::new(grow(axises, classes, left, depth + 1)),
            Box::new(grow(axises, classes, right, depth + 1)),
        )
    } else {
        TrainTree::Leave(data)
    }
}

pub fn train(axises: &[Axis], classes: &[Class], vec: Vec<(DataPoint, Class)>) -> DecisionTree {
    grow(axises, classes, vec, 0).into()
}

fn merge_result(
    mut left: HashMap<Class, Vec<DataPoint>>,
    right: HashMap<Class, Vec<DataPoint>>,
) -> HashMap<Class, Vec<DataPoint>> {
    for (class, mut right_vec) in right {
        match left.get_mut(&class) {
            Some(left_vec) => left_vec.append(&mut right_vec),
            None => {
                left.insert(class, right_vec);
            }
        }
    }
    left
}

pub fn classify(tree: &DecisionTree, data: Vec<DataPoint>) -> HashMap<Class, Vec<DataPoint>> {
    match tree {
        DecisionTree::Branch(c, left_tree, right_tree) => {
            let (left_data, right_data) = c.classify(&data);
            let left = classify(left_tree, left_data);
            let right = classify(right_tree, right_data);
            merge_result(left, right)
        }
        DecisionTree::Leave(class) => {
            let mut result = HashMap::new();
            result.insert(*class, data);
            result
        }
    }
}
