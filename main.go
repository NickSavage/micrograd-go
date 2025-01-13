package main

import (
	"fmt"
)

type Value struct {
	Data     float64
	Grad     float64
	Children []*Value
	Op       string
	Label    string
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
