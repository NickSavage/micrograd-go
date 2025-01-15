package main

import (
	"fmt"
	"math"
	"math/rand/v2"
)

type Value struct {
	Data     float64
	Grad     float64
	Children []*Value
	Op       string
	Label    string
}

type Neuron struct {
	W []*Value
	B *Value
}

type Layer struct {
	Neurons []*Neuron
}

type MLP struct {
	Layers []*Layer
}

func NewNeuron(numInputs int) *Neuron {
	w := make([]*Value, numInputs)
	for i := range w {
		w[i] = &Value{
			Data:     rand.Float64()*2 - 1,
			Grad:     0.0,
			Children: nil,
			Op:       "",
			Label:    fmt.Sprintf("w%d", i),
		}
	}
	b := &Value{
		Data:     rand.Float64()*2 - 1,
		Grad:     0.0,
		Children: nil,
		Op:       "",
		Label:    "b",
	}
	return &Neuron{W: w, B: b}
}

func (n *Neuron) Print() {
	fmt.Printf("Neuron(W: %v, B: %v)\n", n.W, n.B)
}

func (n *Neuron) callValue(x []*Value) *Value {
	sum := n.B.Data
	for i, w := range n.W {
		sum += w.Data * x[i].Data
	}
	out := Tanh(&Value{Data: sum, Grad: 0.0, Children: nil, Op: "", Label: "sum"})
	return out
}

func (n *Neuron) Call(x []float64) float64 {
	values := make([]*Value, len(x))
	for i, v := range x {
		values[i] = &Value{Data: v, Grad: 0.0, Children: nil, Op: "", Label: fmt.Sprintf("x%d", i)}
	}
	return n.callValue(values).Data
}

func NewLayer(numInputs, numOutputs int) *Layer {
	neurons := make([]*Neuron, numOutputs)
	for i := range neurons {
		neurons[i] = NewNeuron(numInputs)
	}
	return &Layer{Neurons: neurons}
}

func (l *Layer) Print() {
	for _, neuron := range l.Neurons {
		neuron.Print()
	}
}

func (l *Layer) callValue(x []*Value) []*Value {
	outputs := make([]*Value, len(l.Neurons))
	for i, neuron := range l.Neurons {
		outputs[i] = neuron.callValue(x)
	}
	return outputs
}

func (l *Layer) Call(x []float64) []float64 {
	values := make([]*Value, len(x))
	for i, v := range x {
		values[i] = &Value{Data: v, Grad: 0.0, Children: nil, Op: "", Label: fmt.Sprintf("x%d", i)}
	}
	outputs := l.callValue(values)
	result := make([]float64, len(outputs))
	for i, v := range outputs {
		result[i] = v.Data
	}
	return result
}

func NewMLP(numInputs int, layerSizes []int) *MLP {
	layers := make([]*Layer, len(layerSizes))

	// current size represents the number of inputs for the next layer
	currentSize := numInputs

	for i, outputSize := range layerSizes {
		layers[i] = NewLayer(currentSize, outputSize)
		currentSize = outputSize
	}

	return &MLP{Layers: layers}
}

func (m *MLP) Print() {
	for _, layer := range m.Layers {
		layer.Print()
	}
}

func (m *MLP) callValue(x []*Value) []*Value {
	for _, layer := range m.Layers {
		x = layer.callValue(x)
	}
	return x
}

func (m *MLP) Call(x []float64) []float64 {
	values := make([]*Value, len(x))
	for i, v := range x {
		values[i] = &Value{Data: v, Grad: 0.0, Children: nil, Op: "", Label: fmt.Sprintf("x%d", i)}
	}
	outputs := m.callValue(values)
	result := make([]float64, len(outputs))
	for i, v := range outputs {
		result[i] = v.Data
	}
	return result
}

func (v *Value) PrintGraph(prefix string, isLast bool) {
	// First line prefix
	firstPrefix := prefix
	if isLast {
		firstPrefix += "└── "
	} else {
		firstPrefix += "├── "
	}

	// Child prefix
	childPrefix := prefix
	if isLast {
		childPrefix += "    "
	} else {
		childPrefix += "│   "
	}

	// Print current node
	opStr := ""
	if v.Op != "" {
		opStr = fmt.Sprintf(" (%s)", v.Op)
	}
	fmt.Printf("%sValue(%s: %.4f, grad=%.4f)%s\n", firstPrefix, v.Label, v.Data, v.Grad, opStr)

	// Print children
	for i, child := range v.Children {
		isLastChild := i == len(v.Children)-1
		child.PrintGraph(childPrefix, isLastChild)
	}
}

func (v *Value) Print() {
	v.PrintGraph("", true)
}

func (v *Value) String() string {
	return fmt.Sprintf("Value(Data: %v, Grad: %v)", v.Data, v.Grad)
}

func Add(a, b *Value) *Value {
	return &Value{Data: a.Data + b.Data, Grad: 0.0, Children: []*Value{a, b}, Op: "+"}
}

func Mul(a, b *Value) *Value {
	return &Value{Data: a.Data * b.Data, Grad: 0.0, Children: []*Value{a, b}, Op: "*"}
}

func Tanh(a *Value) *Value {
	return &Value{Data: math.Tanh(a.Data), Grad: 0.0, Children: []*Value{a}, Op: "tanh"}
}

func (v *Value) backward() {
	switch v.Op {
	case "+":
		v.Children[0].Grad += v.Grad
		v.Children[1].Grad += v.Grad
	case "*":
		v.Children[0].Grad += v.Grad * v.Children[1].Data
		v.Children[1].Grad += v.Grad * v.Children[0].Data
	case "tanh":
		v.Children[0].Grad += v.Grad * (1 - math.Pow(v.Data, 2))
	}
}

func (v *Value) Backward() {
	// Build nodes in topological order
	var topo []*Value
	visited := make(map[*Value]bool)

	var buildTopo func(v *Value)
	buildTopo = func(v *Value) {
		if visited[v] {
			return
		}
		visited[v] = true
		for _, child := range v.Children {
			buildTopo(child)
		}
		topo = append(topo, v)
	}
	buildTopo(v)

	// Initialize gradient of root to 1.0
	v.Grad = 1.0

	// Backpropagate in reverse topological order
	for i := len(topo) - 1; i >= 0; i-- {
		topo[i].backward()
	}
}

func main() {
	m := NewMLP(3, []int{3, 4, 4, 1})

	inputs := [][]float64{
		{2.0, 3.0, -1.0},
		{3.0, -1.0, 0.5},
		{0.5, 1.0, 1.0},
		{1.0, 1.0, -1.0},
	}
	//targets := []float64{1.0, -1.0, -1.0, 1.0}

	fmt.Println("Processing inputs:")
	for i, input := range inputs {
		output := m.Call(input)
		fmt.Printf("Input %d: %v -> Output: %v\n", i+1, input, output)
	}
}
