package main

import (
	"fmt"
	"math"
	"testing"

	"github.com/NickSavage/micrograd-go/nn"
)

func TestBasicAdd(t *testing.T) {
	a := &nn.Value{Data: 1.0, Grad: 0.0}
	b := &nn.Value{Data: 2.0, Grad: 0.0}
	c := nn.Add(a, b)

	fmt.Println(c.Data)
	if c.Data != 3.0 {
		t.Errorf("Expected 3.0, got %f", c.Data)
	}
}

func TestBasicMul(t *testing.T) {
	a := &nn.Value{Data: 2.0, Grad: 0.0}
	b := &nn.Value{Data: 3.0, Grad: 0.0}
	c := nn.Mul(a, b)

	fmt.Println(c.Data)
	if c.Data != 6.0 {
		t.Errorf("Expected 6.0, got %f", c.Data)
	}
}

func TestBasicMulAdd(t *testing.T) {
	a := &nn.Value{Data: 2.0, Grad: 0.0, Label: "a"}
	b := &nn.Value{Data: 3.0, Grad: 0.0, Label: "b"}
	c := nn.Mul(a, b)
	c.Label = "c"
	d := nn.Add(c, a)
	d.Label = "d"
	d.Print()
	if d.Data != 8.0 {
		t.Errorf("Expected 8.0, got %f", d.Data)
	}
}

func TestBasicGrad(t *testing.T) {

	a := &nn.Value{Data: 2.0, Grad: 0.0, Label: "a"}
	b := &nn.Value{Data: -3.0, Grad: 0.0, Label: "b"}
	c := &nn.Value{Data: 10.0, Grad: 0.0, Label: "c"}

	e := nn.Mul(a, b)
	e.Label = "e"

	d := nn.Add(e, c)
	d.Label = "d"

	f := &nn.Value{Data: -2.0, Grad: 0.0, Label: "f"}
	L := nn.Mul(d, f)
	L.Label = "L"
	L.Grad = 1.0
	L.Backward()

	L.Print()
	if L.Data != -8 {
		t.Errorf("Expected -8.0, got %f", L.Data)
	}
	if a.Grad != 6.0 {
		t.Errorf("Expected 6.0, got %f", L.Grad)
	}

}

func TestTanh(t *testing.T) {
	const epsilon = 1e-6

	a := &nn.Value{Data: 1.0, Grad: 0.0}
	b := nn.Tanh(a)
	b.Print()

	if diff := math.Abs(b.Data - math.Tanh(1.0)); diff > epsilon {
		t.Errorf("Expected tanh(1.0) ≈ %f, got %f, diff: %f", math.Tanh(1.0), b.Data, diff)
	}

	b.Grad = 1.0
	b.Backward()
	b.Print()

	expectedGrad := 0.419974
	if diff := math.Abs(a.Grad - expectedGrad); diff > epsilon {
		t.Errorf("Expected gradient ≈ %f, got %f, diff: %f", expectedGrad, a.Grad, diff)
	}
}

func TestSimplifiedGrad(t *testing.T) {
	// Create input values
	x1 := &nn.Value{Data: 2.0, Grad: 0.0, Label: "x1"}
	x2 := &nn.Value{Data: 0.5, Grad: 0.0, Label: "x2"}

	// Create weights
	w1 := &nn.Value{Data: -3.0, Grad: 0.0, Label: "w1"}
	w2 := &nn.Value{Data: 1.0, Grad: 0.0, Label: "w2"}

	// Create bias
	b := &nn.Value{Data: 6.8, Grad: 0.0, Label: "b"}

	// Forward pass - computing: tanh(x1*w1 + x2*w2 + b)
	mul1 := nn.Mul(x1, w1)
	mul1.Label = "x1*w1"

	mul2 := nn.Mul(x2, w2)
	mul2.Label = "x2*w2"

	add1 := nn.Add(mul1, mul2)
	add1.Label = "sum"

	add2 := nn.Add(add1, b)
	add2.Label = "pre_act"

	// Final output with activation
	output := nn.Tanh(add2)
	output.Label = "output"

	// Print forward pass result
	fmt.Println("\n=== Forward Pass ===")
	output.Print()

	// Backward pass
	output.Grad = 1.0
	output.Backward()

	fmt.Println("\n=== After Backward Pass ===")
	output.Print()

	// Test with tolerance
	const epsilon = 1e-6

	// Verify forward pass
	expectedOutput := math.Tanh(2.0*-3.0 + 0.5*1.0 + 6.8)
	if diff := math.Abs(output.Data - expectedOutput); diff > epsilon {
		t.Errorf("Forward pass: Expected ≈ %f, got %f, diff: %f",
			expectedOutput, output.Data, diff)
	}

	// update the following with this: x1.grad tensor(-0.7723)
	// x2.grad tensor(0.2574)
	// w1.grad tensor(0.5149)
	// w2.grad tensor(0.1287)
	// b.grad tensor(0.2574)
	// Verify key gradients
	expectedGrads := map[string]float64{
		"x1": -0.7723, // ∂output/∂x1
		"w1": 0.5149,  // ∂output/∂w1
		"x2": 0.2574,  // ∂output/∂x2
		"b":  0.2574,  // ∂output/∂b
	}

	gradients := map[string]*nn.Value{
		"x1": x1,
		"w1": w1,
		"x2": x2,
		"b":  b,
	}

	for label, expectedGrad := range expectedGrads {
		value := gradients[label]
		if diff := math.Abs(value.Grad - expectedGrad); diff > 0.001 {
			t.Errorf("Gradient for %s: Expected ≈ %f, got %f, diff: %f",
				label, expectedGrad, value.Grad, diff)
		}
	}
}

func TestGradSameComponent(t *testing.T) {
	a := &nn.Value{Data: 2.0, Grad: 0.0, Label: "a"}
	b := nn.Add(a, a)
	b.Print()

	b.Grad = 1.0
	b.Backward()
	b.Print()

	if a.Grad != 2.0 {
		t.Errorf("Expected 2.0, got %f", a.Grad)
	}
}

func TestNeuron(t *testing.T) {
	neuron := nn.NewNeuron(2)
	neuron.Print()
	if len(neuron.W) != 2 {
		t.Errorf("Expected 2 weights, got %d", len(neuron.W))
	}
	if neuron.B == nil {
		t.Errorf("Expected bias, got nil")
	}
}
