(* ::Package:: *)

(************************************************************************)
(* This file was generated automatically by the Mathematica front end.  *)
(* It contains Initialization cells from a Notebook file, which         *)
(* typically will have the same name as this file except ending in      *)
(* ".nb" instead of ".m".                                               *)
(*                                                                      *)
(* This file is intended to be loaded into the Mathematica kernel using *)
(* the package loading commands Get or Needs.  Doing so is equivalent   *)
(* to using the Evaluate Initialization Cells menu command in the front *)
(* end.                                                                 *)
(*                                                                      *)
(* DO NOT EDIT THIS FILE.  This entire file is regenerated              *)
(* automatically each time the parent Notebook file is saved in the     *)
(* Mathematica front end.  Any changes you make to this file will be    *)
(* overwritten.                                                         *)
(************************************************************************)



BeginPackage["dftinput`"]
Unprotect@@Names["dftinput`*"];
ClearAll@@Names["dftinput`*"];


gnrbasis::usage="gnrbasis[rings] returns the coordinates of the carbon atoms contained in a single slice of a zigzag graphene nanoribbon with a given width (number of rings)."
gnrbasiswithh::usage="gnrbasiswithh[rings] returns the coordinates of the carbon and hydrogen atoms contained in a single slice of a hydrogen-passivated zigzag graphene nanoribbon with a given width (number of rings). The last two coordinates are those of the hydrogen atoms."
generategnrfrombasis::usage="generategnrfrombasis[basis,ribbonlength] returns the coordinates of the atoms in a zigzag GNR which contains ribbonlength copies of the given basis (e.g. created with gnrbasis[])."
PointBetweenAtoms2D::usage="PointBetweenAtoms2D[points,latticevecs,neighbourthreshold] returns a list of the points at the half distance between neighbouring atoms. The crystal lattice is given by points and latticevecs. Two points are considered neighbours if their distance is smaller than or equal to neighbourthreshold (e.g. GrapheneLatticeConstant)."
AddZeroInThirdDimension::usage ="AddZeroInThirdDimension[list] adds a third column with zeroes to a list of 2D-Vectors."
GrapheneLatticeConstant::usage="Lattice constant of Graphene."
AppendGNRInitialGuess::usage="AppendGNRInitialGuess[gnrwidth,wannier90file] appends the initial guesses (the \"begin projections\" section) for the Wannier orbitals of a zigzag-GNR with a width of gnrwidth rings to the file wannier90file (e.g. wannier90.win). The atoms are assumed at the positions generated by GenerateGNRPOSCAR[]. The orbital number num_wann is also appended."
AppendGNRhydrogenInitialGuess::usage="AppendGNRhydrogenInitialGuess[gnrwidth,wannier90file] appends the initial guesses (the \"begin projections\" section) for the Wannier orbitals of a zigzag-GNR with hydrogen-passivated edges with a width of gnrwidth rings to the file wannier90file (e.g. wannier90.win). The atoms are assumed at the positions generated by GenerateGNRhydrogenPOSCAR[]. The orbital number num_wann is also appended."
GenerateGNRPOSCAR::usage="GenerateGNRPOSCAR[gnrwidth,yspacing,zspacing,POSCARfile] generates a POSCAR file describing a GNR with gnrwidth rings in width. yspacing and zspacing  is the additional spacing in the box (e.g. 20). A list of the positions of the orbitals is returned."
GenerateGNRhydrogenPOSCAR::usage="GenerateGNRhydrogenPOSCAR[gnrwidth,yspacing,zspacing,POSCARfile] generates a POSCAR file describing a zigzag-GNR with hydrogen-passivated edges with gnrwidth rings in width. yspacing and zspacing  is the additional spacing in the box (e.g. 20). The file contains appropriate parameters (except for acenes) for a relaxation (with NSW and IBRION parameters in the INCAR file); the outmost two carbon atoms (on each side) and the hydrogen atom are being relaxed. The last two coordinates are those of the hydrogen atoms (so the POTCAR file has to contain the carbon first, then the hydrogen). A list of the positions of the orbitals is returned."
Begin["`Private`"]


length=2.461/Sqrt[3];
hydrogendistance=1.1;
GrapheneLatticeConstant=length;
a=length{Sqrt[3]/2,3/2};
b=length{-Sqrt[3]/2,3/2};


unit=({
 {1./3, 1./3},
 {2./3, 2./3}
});
kpath=({
 {0, 0},
 {1./3, -1./3},
 {1./2, 0}
});
kpath2=({
 {0, 0},
 {0.5, 0}
});


unitWithShift[ashift_,bshift_]:=Transpose[{unit[[All,1]]+ashift,unit[[All,1]]+bshift}];
unitWithShift[shifts_]:=Flatten[Apply[unitWithShift,shifts,1],1];
DirectToCartesian[points_]:=#[[1]] a+#[[2]] b&/@points;
CreateAllNumbersInBaseWithLength[base_,length_]:=IntegerDigits[#,base,length]&/@Range[0,base^length-1];
AddZeroInThirdDimension[list_]:=Transpose[Flatten[{Transpose[list],{0.&/@list}},1]] 


ExportListToFile[file_,list_]:=Export[file,list,"Table"];
ExportListToFileWithCommas[file_,list_]:=Export[file,list,"Table","FieldSeparators"->","];
ExportListToFileNoSep[file_,list_]:=Export[file,list,"Table","FieldSeparators"->""];


gnrbasis[rings_]:=Rest[Drop[DirectToCartesian[unitWithShift[Prepend[Flatten[Table[{{n,n-1},{n,n}},{n,Ceiling[(rings+1)/2]}],1],{0,0}]]],If[EvenQ[rings],-3,-1]]];


gnrbasiswithh[rings_]:=Module[{gnrorig},
gnrorig=gnrbasis[rings];
Join[gnrorig,
{First[gnrorig]-{0.,hydrogendistance},
Last[gnrorig]+{0.,hydrogendistance}
}]
];


generategnrfrombasis[basis_,ribbonlength_]:=generategnrfrombasis[length Sqrt[3],basis,ribbonlength];
generategnrfrombasis[latticeconst_,basis_,ribbonlength_]:=Table[basis+n ConstantArray[{latticeconst,0},Length[basis]],{n,ribbonlength}];


PointBetweenAtoms2D[points_,latticevecs_,neighbourthreshold_]:=Module[{lotsofpoints,connections,selectedconnections,selectedconnections2,WithinRange},
WithinRange[point_]:=If[latticevecs[[1]].point<Norm[latticevecs[[1]]]^2&&latticevecs[[2]].point<Norm[latticevecs[[2]]]^2&&
latticevecs[[1]].point>=0&&
latticevecs[[2]].point>=0
,True,False];

lotsofpoints=Join[points,
#+latticevecs[[1]]&/@points,
#-latticevecs[[1]]&/@points,
#+latticevecs[[2]]&/@points,
#-latticevecs[[2]]&/@points,
#+latticevecs[[1]]+latticevecs[[2]]&/@points,
#+latticevecs[[1]]-latticevecs[[2]]&/@points,
#-latticevecs[[1]]+latticevecs[[2]]&/@points,
#-latticevecs[[1]]-latticevecs[[2]]&/@points
];

connections=Flatten[Table[{i,j,lotsofpoints[[j ]]-lotsofpoints[[i ]],(lotsofpoints[[i]]+lotsofpoints[[j]])/2},{i,Length[points]},{j,i+1,Length[lotsofpoints]}],1];
selectedconnections=Select[connections,Norm[#[[3]] ]<=neighbourthreshold&];
selectedconnections2=Select[selectedconnections,
WithinRange[#[[4]]]&];
selectedconnections2[[All,4]]
];

PointBetweenAtoms3D[points_,latticevecs_,neighbourthreshold_]:=Module[{lotsofpoints,connections,selectedconnections,selectedconnections2,WithinRange},
WithinRange[point_]:=If[latticevecs[[1]].point<Norm[latticevecs[[1]]]^2&&latticevecs[[2]].point<Norm[latticevecs[[2]]]^2&&
latticevecs[[3]].point<Norm[latticevecs[[3]]]^2&&
latticevecs[[1]].point>=0&&
latticevecs[[2]].point>=0&&
latticevecs[[3]].point>=0
,True,False];

lotsofpoints=Join[points,
#+latticevecs[[1]]&/@points,
#-latticevecs[[1]]&/@points,
#+latticevecs[[2]]&/@points,
#-latticevecs[[2]]&/@points,
#+latticevecs[[3]]&/@points,
#-latticevecs[[3]]&/@points,
#+latticevecs[[1]]+latticevecs[[2]]+latticevecs[[3]]&/@points,
#+latticevecs[[1]]+latticevecs[[2]]-latticevecs[[3]]&/@points,
#+latticevecs[[1]]-latticevecs[[2]]+latticevecs[[3]]&/@points,
#+latticevecs[[1]]-latticevecs[[2]]-latticevecs[[3]]&/@points,
#-latticevecs[[1]]+latticevecs[[2]]+latticevecs[[3]]&/@points,
#-latticevecs[[1]]+latticevecs[[2]]-latticevecs[[3]]&/@points,
#-latticevecs[[1]]-latticevecs[[2]]+latticevecs[[3]]&/@points,
#-latticevecs[[1]]-latticevecs[[2]]-latticevecs[[3]]&/@points
];

connections=Flatten[Table[{i,j,lotsofpoints[[j ]]-lotsofpoints[[i ]],(lotsofpoints[[i]]+lotsofpoints[[j]])/2},{i,Length[points]},{j,i+1,Length[lotsofpoints]}],1];
selectedconnections=Select[connections,Norm[#[[3]] ]<=neighbourthreshold&];
selectedconnections2=Select[selectedconnections,
WithinRange[#[[4]]]&];
selectedconnections2[[All,4]]
];


AppendGNRInitialGuess[gnrwidth_,wannier90file_]:=Module[{betwpoints,initialguess1,initialguess2,initialguess3,NumberToString,basis,wfile,raw1,raw2,raw3},
NumberToString[x_]:=ToString[NumberForm[x,20]];
basis=gnrbasis[gnrwidth];
betwpoints=PointBetweenAtoms2D[basis,{{GrapheneLatticeConstant Sqrt[3],0},{0,10000}},GrapheneLatticeConstant];
raw1=Rest[Most[basis]];
raw2=betwpoints;
raw3={First[basis],Last[basis]};
initialguess1={"c=",#[[1]],",",#[[2]],",",0,":pz"}&/@raw1;
initialguess2={"c=",#[[1]],",",#[[2]],",",0,":s"}&/@raw2;
initialguess3={"c=",#[[1]],",",#[[2]],",",0,":pz,py"}&/@raw3;
wfile=If[FileExistsQ[wannier90file],Import[wannier90file,"Lines"],{}];

Export[wannier90file,Join[wfile,{
"",
"num_wann="<>ToString[Length[initialguess1]+Length[initialguess2]+2Length[initialguess3]],
"",
"begin projections"
},
StringJoin[NumberToString/@#]&/@Join[initialguess1,initialguess2,initialguess3],
{
"end projections"
},
{" "}],"Lines"];
Join[raw1,raw2,raw3]
];

AppendGNRhydrogenInitialGuess[gnrwidth_,wannier90file_]:=Module[{betwpoints,initialguess1,initialguess2,initialguess3,NumberToString,basis,wfile,raw1,raw2,raw3},
NumberToString[x_]:=ToString[NumberForm[x,20]];
basis=gnrbasiswithh[gnrwidth];
betwpoints=PointBetweenAtoms2D[Drop[basis,-2](*only c atoms*),{{GrapheneLatticeConstant Sqrt[3],0},{0,10000}},GrapheneLatticeConstant];
raw1=Drop[basis,-2];
raw2=betwpoints;
raw3=Take[basis,-2];
initialguess1={"c=",#[[1]],",",#[[2]],",",0,":pz"}&/@raw1;
initialguess2={"c=",#[[1]],",",#[[2]],",",0,":s"}&/@raw2;
initialguess3={"c=",#[[1]],",",#[[2]],",",0,":s"}&/@raw3;
wfile=If[FileExistsQ[wannier90file],Import[wannier90file,"Lines"],{}];

Export[wannier90file,Join[wfile,{
"",
"num_wann="<>ToString[Length[initialguess1]+Length[initialguess2]+Length[initialguess3]],
"",
"begin projections"
},
StringJoin[NumberToString/@#]&/@Join[initialguess1,initialguess2,initialguess3],
{
"end projections"
},
{" "}],"Lines"];
Join[raw1,raw2,raw3]
];


GenerateGNRPOSCAR[gnrwidth_,yspacing_,zspacing_,POSCARfile_]:=Module[{gnrrealwidth,basis,filecontent},
basis=gnrbasis[gnrwidth];
gnrrealwidth=Max[#]-Min[#]&[basis[[All,2]] ] ;
filecontent=Join[{
{"GNR Zigzag "<>ToString[gnrwidth]},
{1},
{GrapheneLatticeConstant Sqrt[3],0.0,0.0},
{0.0,gnrrealwidth+yspacing,0.0},(*the lattice vectors should not have the same length*)
{0.0,0.0,zspacing},
{"C"},
{basis//Length},
{"Cartesian"}
},
AddZeroInThirdDimension[basis],
{" "}
];
Export[POSCARfile,filecontent,"Table","FieldSeparators"->" ","TextDelimiters"->""];
];

GenerateGNRhydrogenPOSCAR[gnrwidth_,yspacing_,zspacing_,POSCARfile_]:=Module[{gnrrealwidth,basis,filecontent,relaxationparameters},
basis=gnrbasiswithh[gnrwidth];
relaxationparameters=Join[
{{"T","T","F"},(*Two bottom carbon*)
{"T","T","F"}},
Table[{"F","F","F"},{i,Length[basis]-6}],
{{"T","T","F"},(*Two top carbon*)
{"T","T","F"},
{"T","T","F"},(*Hydrogen*)
{"T","T","F"}}
];
gnrrealwidth=Max[#]-Min[#]&[basis[[All,2]] ] ;
filecontent=Join[{
{"GNR Zigzag Hydrogenated "<>ToString[gnrwidth]},
{1},
{GrapheneLatticeConstant Sqrt[3],0.0,0.0},
{0.0,gnrrealwidth+yspacing,0.0},(*the lattice vectors should not have the same length because vasp assumes a wrong symmetry then*)
{0.0,0.0,zspacing},
{"C","H"},
{Length[basis]-2,2},
{"Selective Dynamics"},
{"Cartesian"}
},
Transpose[Join[Transpose[AddZeroInThirdDimension[basis]],Transpose[relaxationparameters]]],
{" "}
];
Export[POSCARfile,filecontent,"Table","FieldSeparators"->" ","TextDelimiters"->""];
];


End[]
Protect@@Names["dftinput`*"];
EndPackage[]