use std::time::Instant;
use rand::Rng;
use spfunc::gamma::*;
use rand_distr::{Normal, Distribution};
use std::f64::{self, consts};

const STEP_SIZE:f64 = 0.01;
const λ :f64 = 0.3;
const γ :f64 = 0.1;
const N:usize = 10;
const P:usize = 20;
const D:usize = 2;

fn create_x(real_choice:usize, ra:f64, rb:f64) -> [f64; N] {
    let mut x = [0.0; N];
    match real_choice {
        1 => {
            for i in 1..N {
                x[i] = ra * i as f64 + rb;
            }
        },
        2 => {
            for i in 1..N {
                x[i] = ra*(i as f64).powf(2.0) + rb*i as f64;
            }
        },
        _ => {
            let mut input = String::new();
            for i in 1..N {
                std::io::stdin()
                    .read_line(&mut input)
                    .expect("can not read user input");
                x[i] = input.parse().unwrap();
            }
        },
    }
    x
}

fn model_lnr(x:f64, a:f64, b:f64) -> f64 {
    a*x + b
}

fn model_sqr(x:f64, a:f64, b:f64) -> f64 {
    a*x.powf(2.0) + b*x
}

fn obj_func(model_choice:usize, x:[f64; N], a:f64, b:f64) -> f64 {
    let mut sum = 0.0;
    match model_choice {
        1 => {
            for i in 1..N {
                sum += (x[i] - model_lnr(i as f64, a, b)).powf(2.0);
            }
        },
        _ => {
            for i in 1..N {
                sum += (x[i] - model_sqr(i as f64, a, b)).powf(2.0);
            }
        },
    }
    sum/N as f64
}

fn obj_compare(model_choice:usize, x:[f64; N], a:f64, b:f64) {
    match model_choice {
        1 => {
            for i in 1..N {
                println!("predict: {} | real: {}", model_lnr(i as f64, a, b), x[i]);
            }
        },
        _ => {
            for i in 1..N {
                println!("predict: {} | real: {}", model_sqr(i as f64, a, b), x[i]);
            }
        },
    }
}

fn get_gbest(model_choice:usize, real_x:[f64; N], population:[[f64; D]; P], g_best:[f64;D]) -> [f64; D] {
    let mut gbest = g_best;
    for i in 1..P {
        if obj_func(model_choice, real_x, population[i][0], population[i][1]) < obj_func(model_choice, real_x, g_best[0], g_best[1]) {
            gbest = population[i];
        }
    }
    gbest
}

fn flower(mut population:[[f64; D]; P]) -> [[f64; D]; P]{
    let mut rng = rand::thread_rng();
    for i in 0..P {
        for j in 0..D {
            population[i][j] = rng.gen_range(-100.0..100.0);
        }
    }
    population
}

fn pollinate(model_choice:usize, real_x:[f64; N], mut population:[[f64; D]; P]) -> [f64; D] {
    let mut rng = rand::thread_rng();
    let mut g_best = get_gbest(model_choice, real_x, population, [1000.0;D]);
    for epoch in 0..100000 {
        g_best = get_gbest(model_choice, real_x, population, g_best);
        // if epoch%100 == 0 {
        //     println!("epoch: {} | g_best_a: {} | g_best_b: {} | obj: {}", epoch, g_best[0], g_best[1], obj_func(model_choice, real_x, g_best[0], g_best[1]));
        // }
        for i in 1..P {
            let mut temp_population = [0.0;D];
            if rng.gen_range(0.0..1.0) < 0.6 {
                temp_population[0] += γ*levi_func()*(g_best[0] - population[i][0]);
                temp_population[1] += γ*levi_func()*(g_best[1] - population[i][1]);
            } else {
                temp_population[0] += rng.gen_range(0.0..1.0)*(population[rng.gen_range(0..P)][0] - population[rng.gen_range(0..P)][0]);
                temp_population[1] += rng.gen_range(0.0..1.0)*(population[rng.gen_range(0..P)][1] - population[rng.gen_range(0..P)][1]);
            }
            if obj_func(model_choice, real_x, temp_population[0], temp_population[1]) < obj_func(model_choice, real_x, population[i][0], population[i][1]) {
                population[i] = temp_population;
            }
        }
    }
    println!("g_best_a: {} | g_best_b: {} | obj: {}", g_best[0], g_best[1], obj_func(model_choice, real_x, g_best[0], g_best[1]));
    g_best
}


fn levi_func() -> f64 {
    //println!("{}",(λ*gamma(λ)*(consts::PI*λ/2.0).sin()/consts::PI) * (1.0/STEP_SIZE.powf(λ+1.0)));
    let normal = Normal::new(2.0, 3.0).unwrap();
    let sigma = (λ*gamma(λ)*(consts::PI*λ/2.0).sin()/consts::PI) * (1.0/STEP_SIZE.powf(λ+1.0));
    let step = normal.sample(&mut rand::thread_rng()) * sigma / normal.sample(&mut rand::thread_rng()).powf(1.0/λ);
    step
    // levi-function
    // if x <= mu {
    //     return 0.0;
    // }
    // (c / (2.0 * std::f64::consts::PI)).sqrt() * (1.0/(x-mu).powf(3.0/2.0)) * (std::f64::consts::E).powf(-c/(2.0*(x-mu)))
}

fn main() {
    let mut input1 = String::new();

    println!("Input real A constant:");

    std::io::stdin()
        .read_line(&mut input1)
        .expect("can not read user input");

    let real_a : f64 = input1.trim().parse().expect("Input not a float");

    println!("Input real B constant:");

    let mut input2 = String::new();

    std::io::stdin()
        .read_line(&mut input2)
        .expect("can not read user input");
    let real_b : f64 = input2.trim().parse().expect("Input not a float");

    println!("Input Model Choice");

    let mut input = String::new();

    std::io::stdin()
        .read_line(&mut input)
        .expect("can not read user input");
    let model_choice = input.trim().parse().expect("Input not an integer");
    
    let real_x = create_x(model_choice, real_a, real_b); 

    let before = Instant::now();

    let mut pollinators = [[1.0; D]; P];
    pollinators = flower(pollinators);

    let g_minima = pollinate(model_choice, real_x, pollinators);

    println!("Elapsed time: {:.2?}", before.elapsed());
    
    obj_compare(model_choice, real_x, g_minima[0], g_minima[1]);
    std::io::stdin()
        .read_line(&mut input)
        .expect("can not read user input");
}
