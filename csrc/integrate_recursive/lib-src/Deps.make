affirm.ho: affirm.h

argparser_extra.ho: argparser_extra.h
argparser_extra.ho: vec.ho
argparser_extra.ho: affirm.ho
argparser_extra.ho: argparser.ho

argparser.ho: argparser.h
argparser.ho: vec.ho
argparser.ho: affirm.ho

bool.ho: bool.h

fget.ho: fget.h
fget.ho: bool.ho

filefmt.ho: filefmt.h

float_image_color.ho: float_image_color.h
float_image_color.ho: bool.ho
float_image_color.ho: r2.ho
float_image_color.ho: i2.ho
float_image_color.ho: float_image.ho
float_image_color.ho: frgb.ho

float_image_expand_by_one.ho: float_image_expand_by_one.h
float_image_expand_by_one.ho: float_image.ho

float_image.ho: float_image.h
float_image.ho: bool.ho
float_image.ho: ix.ho

float_image_mscale.ho: float_image_mscale.h
float_image_mscale.ho: bool.ho
float_image_mscale.ho: r2.ho
float_image_mscale.ho: float_image.ho

frgb.ho: frgb.h

frgb_ops.ho: frgb_ops.h
frgb_ops.ho: argparser.ho
frgb_ops.ho: bool.ho
frgb_ops.ho: frgb.ho

gausol_print.ho: gausol_print.h
gausol_print.ho: bool.ho

gausol_solve.ho: gausol_solve.h
gausol_solve.ho: bool.ho

gausol_triang.ho: gausol_triang.h
gausol_triang.ho: bool.ho

gauss_bell.ho: gauss_bell.h
gauss_bell.ho: bool.ho

gauss_distr.ho: gauss_distr.h
gauss_distr.ho: bool.ho
gauss_distr.ho: gauss_bell.ho

i2.ho: i2.h
i2.ho: vec.ho
i2.ho: sign.ho

interval.ho: interval.h
interval.ho: bool.ho
interval.ho: jsmath.ho

ix.ho: ix.h
ix.ho: bool.ho
ix.ho: sign.ho
ix.ho: ix_types.ho
ix.ho: ix_reduce.ho

ix_reduce.ho: ix_reduce.h
ix_reduce.ho: ix_types.ho

ix_types.ho: ix_types.h

jsfile.ho: jsfile.h
jsfile.ho: bool.ho

jsmath.ho: jsmath.h

jsprintf.ho: jsprintf.h

jsrandom.ho: jsrandom.h

jsstring.ho: jsstring.h
jsstring.ho: bool.ho

jswsize.ho: jswsize.h

nget.ho: nget.h
nget.ho: bool.ho

pst_argparser.ho: pst_argparser.h
pst_argparser.ho: argparser.ho
pst_argparser.ho: bool.ho
pst_argparser.ho: vec.ho
pst_argparser.ho: r2.ho
pst_argparser.ho: pst_basic.ho

pst_basic.ho: pst_basic.h
pst_basic.ho: float_image.ho
pst_basic.ho: vec.ho
pst_basic.ho: r2.ho
pst_basic.ho: r3.ho
pst_basic.ho: frgb.ho
pst_basic.ho: argparser.ho

pst_cell_map_clear.ho: pst_cell_map_clear.h
pst_cell_map_clear.ho: i2.ho
pst_cell_map_clear.ho: vec.ho
pst_cell_map_clear.ho: float_image.ho

pst_cell_map_shrink.ho: pst_cell_map_shrink.h
pst_cell_map_shrink.ho: bool.ho
pst_cell_map_shrink.ho: float_image.ho

pst_height_map.ho: pst_height_map.h
pst_height_map.ho: float_image.ho

pst_imgsys.ho: pst_imgsys.h
pst_imgsys.ho: bool.ho
pst_imgsys.ho: float_image.ho

pst_imgsys_solve.ho: pst_imgsys_solve.h
pst_imgsys_solve.ho: bool.ho
pst_imgsys_solve.ho: pst_imgsys.ho

pst_integrate.ho: pst_integrate.h
pst_integrate.ho: bool.ho
pst_integrate.ho: r2.ho
pst_integrate.ho: float_image.ho
pst_integrate.ho: pst_imgsys.ho
pst_integrate.ho: pst_imgsys_solve.ho
pst_integrate.ho: pst_slope_map.ho
pst_integrate.ho: pst_height_map.ho

pst_integrate_iterative.ho: pst_integrate_iterative.h
pst_integrate_iterative.ho: bool.ho
pst_integrate_iterative.ho: r2.ho
pst_integrate_iterative.ho: float_image.ho
pst_integrate_iterative.ho: pst_imgsys.ho
pst_integrate_iterative.ho: pst_imgsys_solve.ho
pst_integrate_iterative.ho: pst_slope_map.ho
pst_integrate_iterative.ho: pst_height_map.ho
pst_integrate_iterative.ho: pst_integrate.ho

pst_integrate_recursive.ho: pst_integrate_recursive.h
pst_integrate_recursive.ho: bool.ho
pst_integrate_recursive.ho: r2.ho
pst_integrate_recursive.ho: float_image.ho
pst_integrate_recursive.ho: pst_imgsys.ho
pst_integrate_recursive.ho: pst_imgsys_solve.ho
pst_integrate_recursive.ho: pst_slope_map.ho
pst_integrate_recursive.ho: pst_height_map.ho
pst_integrate_recursive.ho: pst_integrate.ho

pst_interpolate.ho: pst_interpolate.h
pst_interpolate.ho: float_image.ho

pst_map_clear.ho: pst_map_clear.h
pst_map_clear.ho: bool.ho
pst_map_clear.ho: float_image.ho

pst_map_compare.ho: pst_map_compare.h
pst_map_compare.ho: bool.ho
pst_map_compare.ho: float_image.ho

pst_map.ho: pst_map.h
pst_map.ho: vec.ho
pst_map.ho: float_image.ho

pst_map_shift_values.ho: pst_map_shift_values.h
pst_map_shift_values.ho: sign.ho
pst_map_shift_values.ho: float_image.ho

pst_normal_map.ho: pst_normal_map.h
pst_normal_map.ho: float_image.ho
pst_normal_map.ho: r2.ho
pst_normal_map.ho: r3.ho
pst_normal_map.ho: r3x3.ho
pst_normal_map.ho: argparser.ho
pst_normal_map.ho: pst_basic.ho

pst_slope_map.ho: pst_slope_map.h
pst_slope_map.ho: bool.ho
pst_slope_map.ho: r2.ho
pst_slope_map.ho: float_image.ho
pst_slope_map.ho: pst_height_map.ho

pst_vertex_map_clear.ho: pst_vertex_map_clear.h
pst_vertex_map_clear.ho: i2.ho
pst_vertex_map_clear.ho: vec.ho
pst_vertex_map_clear.ho: float_image.ho

pst_vertex_map_expand.ho: pst_vertex_map_expand.h
pst_vertex_map_expand.ho: float_image.ho

pst_vertex_map_shrink.ho: pst_vertex_map_shrink.h
pst_vertex_map_shrink.ho: float_image.ho

r2.ho: r2.h
r2.ho: sign.ho
r2.ho: vec.ho
r2.ho: interval.ho

r3.ho: r3.h
r3.ho: sign.ho
r3.ho: vec.ho
r3.ho: sign.ho
r3.ho: interval.ho

r3x3.ho: r3x3.h
r3x3.ho: sign.ho
r3x3.ho: r3.ho

ref.ho: ref.h

rmxn.ho: rmxn.h
rmxn.ho: rn.ho

rn.ho: rn.h
rn.ho: bool.ho

sample_conv_gamma.ho: sample_conv_gamma.h
sample_conv_gamma.ho: bool.ho

sample_conv.ho: sample_conv.h
sample_conv.ho: bool.ho

sign_get.ho: sign_get.h
sign_get.ho: sign.ho

sign.ho: sign.h

vec.ho: vec.h
vec.ho: bool.ho
vec.ho: ref.ho

wt_table_binomial.ho: wt_table_binomial.h
wt_table_binomial.ho: vec.ho
wt_table_binomial.ho: bool.ho
wt_table_binomial.ho: argparser.ho

wt_table.ho: wt_table.h
wt_table.ho: vec.ho
wt_table.ho: bool.ho

wt_table_hann.ho: wt_table_hann.h
wt_table_hann.ho: vec.ho
wt_table_hann.ho: bool.ho
wt_table_hann.ho: argparser.ho

affirm.o: affirm.c
affirm.o: affirm.ho

argparser.o: argparser.c
argparser.o: vec.ho
argparser.o: affirm.ho
argparser.o: jswsize.ho
argparser.o: argparser.ho
argparser.o: argparser_extra.ho

argparser_extra.o: argparser_extra.c
argparser_extra.o: affirm.ho
argparser_extra.o: jswsize.ho
argparser_extra.o: argparser.ho
argparser_extra.o: argparser_extra.ho

bool.o: bool.c
bool.o: bool.ho

fget.o: fget.c
fget.o: jsstring.ho
fget.o: affirm.ho
fget.o: bool.ho
fget.o: fget.ho

filefmt.o: filefmt.c
filefmt.o: filefmt.ho
filefmt.o: fget.ho
filefmt.o: jsstring.ho
filefmt.o: jsprintf.ho
filefmt.o: affirm.ho
filefmt.o: vec.ho

float_image.o: float_image.c
float_image.o: sample_conv.ho
float_image.o: sample_conv_gamma.ho
float_image.o: frgb.ho
float_image.o: frgb_ops.ho
float_image.o: float_image_color.ho
float_image.o: ix.ho
float_image.o: jswsize.ho
float_image.o: jsfile.ho
float_image.o: filefmt.ho
float_image.o: nget.ho
float_image.o: fget.ho
float_image.o: affirm.ho
float_image.o: bool.ho
float_image.o: float_image.ho

float_image_color.o: float_image_color.c
float_image_color.o: bool.ho
float_image_color.o: r2.ho
float_image_color.o: i2.ho
float_image_color.o: jsmath.ho
float_image_color.o: affirm.ho
float_image_color.o: float_image.ho
float_image_color.o: float_image_color.ho
float_image_color.o: frgb.ho
float_image_color.o: frgb_ops.ho

float_image_expand_by_one.o: float_image_expand_by_one.c
float_image_expand_by_one.o: affirm.ho
float_image_expand_by_one.o: float_image.ho
float_image_expand_by_one.o: float_image_expand_by_one.ho

float_image_mscale.o: float_image_mscale.c
float_image_mscale.o: bool.ho
float_image_mscale.o: affirm.ho
float_image_mscale.o: jsprintf.ho
float_image_mscale.o: r2.ho
float_image_mscale.o: jsfile.ho
float_image_mscale.o: wt_table.ho
float_image_mscale.o: wt_table_binomial.ho
float_image_mscale.o: float_image.ho
float_image_mscale.o: float_image_mscale.ho

frgb.o: frgb.c
frgb.o: frgb.ho

frgb_ops.o: frgb_ops.c
frgb_ops.o: frgb_ops.ho
frgb_ops.o: sample_conv.ho
frgb_ops.o: sample_conv_gamma.ho
frgb_ops.o: argparser.ho
frgb_ops.o: frgb.ho
frgb_ops.o: bool.ho
frgb_ops.o: fget.ho

gausol_print.o: gausol_print.c
gausol_print.o: affirm.ho
gausol_print.o: jsprintf.ho
gausol_print.o: bool.ho
gausol_print.o: jsmath.ho
gausol_print.o: gausol_print.ho

gausol_solve.o: gausol_solve.c
gausol_solve.o: affirm.ho
gausol_solve.o: gausol_triang.ho
gausol_solve.o: gausol_print.ho
gausol_solve.o: gausol_solve.ho

gausol_triang.o: gausol_triang.c
gausol_triang.o: bool.ho
gausol_triang.o: affirm.ho
gausol_triang.o: gausol_triang.ho

gauss_bell.o: gauss_bell.c
gauss_bell.o: bool.ho
gauss_bell.o: jsmath.ho
gauss_bell.o: affirm.ho
gauss_bell.o: gauss_bell.ho

gauss_distr.o: gauss_distr.c
gauss_distr.o: bool.ho
gauss_distr.o: jsmath.ho
gauss_distr.o: affirm.ho
gauss_distr.o: gauss_bell.ho
gauss_distr.o: gauss_distr.ho

i2.o: i2.c
i2.o: jsrandom.ho
i2.o: affirm.ho
i2.o: vec.ho
i2.o: sign.ho
i2.o: sign_get.ho
i2.o: i2.ho

interval.o: interval.c
interval.o: bool.ho
interval.o: jsmath.ho
interval.o: affirm.ho
interval.o: interval.ho

ix.o: ix.c
ix.o: affirm.ho
ix.o: ix_types.ho
ix.o: ix_reduce.ho
ix.o: ix.ho

ix_reduce.o: ix_reduce.c
ix_reduce.o: affirm.ho
ix_reduce.o: bool.ho
ix_reduce.o: ix_types.ho
ix_reduce.o: ix_reduce.ho

jsfile.o: jsfile.c
jsfile.o: jsfile.ho
jsfile.o: jsprintf.ho
jsfile.o: affirm.ho

jsmath.o: jsmath.c
jsmath.o: jsmath.ho
jsmath.o: affirm.ho

jsprintf.o: jsprintf.c
jsprintf.o: affirm.ho
jsprintf.o: jsprintf.ho

jsrandom.o: jsrandom.c
jsrandom.o: affirm.ho
jsrandom.o: jsrandom.ho

jsstring.o: jsstring.c
jsstring.o: affirm.ho
jsstring.o: jsprintf.ho
jsstring.o: bool.ho
jsstring.o: jsstring.ho

nget.o: nget.c
nget.o: nget.ho
nget.o: fget.ho
nget.o: affirm.ho

pst_argparser.o: pst_argparser.c
pst_argparser.o: float_image.ho
pst_argparser.o: r2.ho
pst_argparser.o: vec.ho
pst_argparser.o: affirm.ho
pst_argparser.o: argparser.ho
pst_argparser.o: pst_argparser.ho
pst_argparser.o: pst_basic.ho

pst_basic.o: pst_basic.c
pst_basic.o: float_image.ho
pst_basic.o: vec.ho
pst_basic.o: r2.ho
pst_basic.o: r3.ho
pst_basic.o: argparser.ho
pst_basic.o: pst_basic.ho

pst_cell_map_clear.o: pst_cell_map_clear.c
pst_cell_map_clear.o: bool.ho
pst_cell_map_clear.o: i2.ho
pst_cell_map_clear.o: vec.ho
pst_cell_map_clear.o: affirm.ho
pst_cell_map_clear.o: float_image.ho
pst_cell_map_clear.o: pst_basic.ho
pst_cell_map_clear.o: pst_map_clear.ho
pst_cell_map_clear.o: pst_cell_map_clear.ho

pst_cell_map_shrink.o: pst_cell_map_shrink.c
pst_cell_map_shrink.o: bool.ho
pst_cell_map_shrink.o: affirm.ho
pst_cell_map_shrink.o: float_image.ho
pst_cell_map_shrink.o: pst_basic.ho
pst_cell_map_shrink.o: pst_cell_map_shrink.ho

pst_height_map.o: pst_height_map.c
pst_height_map.o: bool.ho
pst_height_map.o: affirm.ho
pst_height_map.o: jsprintf.ho
pst_height_map.o: float_image.ho
pst_height_map.o: pst_map.ho
pst_height_map.o: pst_vertex_map_shrink.ho
pst_height_map.o: pst_vertex_map_expand.ho
pst_height_map.o: pst_map_compare.ho
pst_height_map.o: pst_height_map.ho

pst_imgsys.o: pst_imgsys.c
pst_imgsys.o: bool.ho
pst_imgsys.o: affirm.ho
pst_imgsys.o: jsfile.ho
pst_imgsys.o: float_image.ho
pst_imgsys.o: float_image_mscale.ho
pst_imgsys.o: filefmt.ho
pst_imgsys.o: pst_imgsys.ho

pst_imgsys_solve.o: pst_imgsys_solve.c
pst_imgsys_solve.o: rn.ho
pst_imgsys_solve.o: affirm.ho
pst_imgsys_solve.o: jsmath.ho
pst_imgsys_solve.o: pst_imgsys.ho
pst_imgsys_solve.o: pst_imgsys_solve.ho

pst_integrate.o: pst_integrate.c
pst_integrate.o: float_image.ho
pst_integrate.o: float_image_mscale.ho
pst_integrate.o: jsfile.ho
pst_integrate.o: r2.ho
pst_integrate.o: jswsize.ho
pst_integrate.o: rn.ho
pst_integrate.o: pst_basic.ho
pst_integrate.o: pst_imgsys.ho
pst_integrate.o: pst_imgsys_solve.ho
pst_integrate.o: pst_interpolate.ho
pst_integrate.o: pst_slope_map.ho
pst_integrate.o: pst_height_map.ho
pst_integrate.o: pst_integrate.ho

pst_integrate_iterative.o: pst_integrate_iterative.c
pst_integrate_iterative.o: float_image.ho
pst_integrate_iterative.o: float_image_mscale.ho
pst_integrate_iterative.o: jsfile.ho
pst_integrate_iterative.o: r2.ho
pst_integrate_iterative.o: jswsize.ho
pst_integrate_iterative.o: rn.ho
pst_integrate_iterative.o: pst_basic.ho
pst_integrate_iterative.o: pst_imgsys.ho
pst_integrate_iterative.o: pst_imgsys_solve.ho
pst_integrate_iterative.o: pst_interpolate.ho
pst_integrate_iterative.o: pst_slope_map.ho
pst_integrate_iterative.o: pst_height_map.ho
pst_integrate_iterative.o: pst_integrate.ho
pst_integrate_iterative.o: pst_integrate_iterative.ho

pst_integrate_recursive.o: pst_integrate_recursive.c
pst_integrate_recursive.o: float_image.ho
pst_integrate_recursive.o: float_image_mscale.ho
pst_integrate_recursive.o: jsfile.ho
pst_integrate_recursive.o: r2.ho
pst_integrate_recursive.o: jswsize.ho
pst_integrate_recursive.o: rn.ho
pst_integrate_recursive.o: pst_basic.ho
pst_integrate_recursive.o: pst_imgsys.ho
pst_integrate_recursive.o: pst_imgsys_solve.ho
pst_integrate_recursive.o: pst_interpolate.ho
pst_integrate_recursive.o: pst_slope_map.ho
pst_integrate_recursive.o: pst_height_map.ho
pst_integrate_recursive.o: pst_integrate.ho
pst_integrate_recursive.o: pst_integrate_iterative.ho
pst_integrate_recursive.o: pst_integrate_recursive.ho

pst_interpolate.o: pst_interpolate.c
pst_interpolate.o: affirm.ho
pst_interpolate.o: float_image.ho
pst_interpolate.o: pst_interpolate.ho

pst_map.o: pst_map.c
pst_map.o: vec.ho
pst_map.o: float_image.ho
pst_map.o: pst_basic.ho
pst_map.o: pst_interpolate.ho
pst_map.o: pst_map.ho

pst_map_clear.o: pst_map_clear.c
pst_map_clear.o: bool.ho
pst_map_clear.o: affirm.ho
pst_map_clear.o: float_image.ho
pst_map_clear.o: pst_map_clear.ho

pst_map_compare.o: pst_map_compare.c
pst_map_compare.o: bool.ho
pst_map_compare.o: affirm.ho
pst_map_compare.o: float_image.ho
pst_map_compare.o: float_image_mscale.ho
pst_map_compare.o: pst_map_shift_values.ho
pst_map_compare.o: pst_map_compare.ho

pst_map_shift_values.o: pst_map_shift_values.c
pst_map_shift_values.o: bool.ho
pst_map_shift_values.o: sign.ho
pst_map_shift_values.o: affirm.ho
pst_map_shift_values.o: float_image.ho
pst_map_shift_values.o: pst_basic.ho
pst_map_shift_values.o: pst_map_shift_values.ho

pst_normal_map.o: pst_normal_map.c
pst_normal_map.o: float_image.ho
pst_normal_map.o: r2.ho
pst_normal_map.o: r3.ho
pst_normal_map.o: r3x3.ho
pst_normal_map.o: jsrandom.ho
pst_normal_map.o: affirm.ho
pst_normal_map.o: pst_normal_map.ho
pst_normal_map.o: pst_slope_map.ho

pst_slope_map.o: pst_slope_map.c
pst_slope_map.o: bool.ho
pst_slope_map.o: affirm.ho
pst_slope_map.o: float_image.ho
pst_slope_map.o: jsrandom.ho
pst_slope_map.o: pst_basic.ho
pst_slope_map.o: pst_map.ho
pst_slope_map.o: pst_interpolate.ho
pst_slope_map.o: pst_cell_map_shrink.ho
pst_slope_map.o: pst_slope_map.ho

pst_vertex_map_clear.o: pst_vertex_map_clear.c
pst_vertex_map_clear.o: bool.ho
pst_vertex_map_clear.o: i2.ho
pst_vertex_map_clear.o: vec.ho
pst_vertex_map_clear.o: affirm.ho
pst_vertex_map_clear.o: float_image.ho
pst_vertex_map_clear.o: pst_basic.ho
pst_vertex_map_clear.o: pst_map_clear.ho
pst_vertex_map_clear.o: pst_vertex_map_clear.ho

pst_vertex_map_expand.o: pst_vertex_map_expand.c
pst_vertex_map_expand.o: bool.ho
pst_vertex_map_expand.o: affirm.ho
pst_vertex_map_expand.o: float_image.ho
pst_vertex_map_expand.o: pst_basic.ho
pst_vertex_map_expand.o: pst_vertex_map_expand.ho

pst_vertex_map_shrink.o: pst_vertex_map_shrink.c
pst_vertex_map_shrink.o: bool.ho
pst_vertex_map_shrink.o: affirm.ho
pst_vertex_map_shrink.o: float_image.ho
pst_vertex_map_shrink.o: pst_basic.ho
pst_vertex_map_shrink.o: pst_vertex_map_shrink.ho

r2.o: r2.c
r2.o: r2.ho
r2.o: jsrandom.ho
r2.o: interval.ho
r2.o: affirm.ho
r2.o: sign.ho
r2.o: sign_get.ho
r2.o: rn.ho
r2.o: vec.ho

r3.o: r3.c
r3.o: r3.ho
r3.o: jsrandom.ho
r3.o: affirm.ho
r3.o: sign.ho
r3.o: rn.ho
r3.o: vec.ho

r3x3.o: r3x3.c
r3x3.o: r3.ho
r3x3.o: affirm.ho
r3x3.o: rmxn.ho
r3x3.o: sign.ho
r3x3.o: jsrandom.ho
r3x3.o: r3x3.ho

rmxn.o: rmxn.c
rmxn.o: rn.ho
rmxn.o: bool.ho
rmxn.o: jsmath.ho
rmxn.o: jsrandom.ho
rmxn.o: affirm.ho
rmxn.o: gausol_triang.ho
rmxn.o: gausol_solve.ho
rmxn.o: rmxn.ho

rn.o: rn.c
rn.o: rn.ho
rn.o: rmxn.ho
rn.o: bool.ho
rn.o: jsrandom.ho
rn.o: jsmath.ho
rn.o: affirm.ho
rn.o: gausol_triang.ho
rn.o: gausol_solve.ho

sample_conv.o: sample_conv.c
sample_conv.o: affirm.ho
sample_conv.o: bool.ho
sample_conv.o: jsmath.ho
sample_conv.o: sample_conv.ho

sample_conv_gamma.o: sample_conv_gamma.c
sample_conv_gamma.o: affirm.ho
sample_conv_gamma.o: bool.ho
sample_conv_gamma.o: jsmath.ho
sample_conv_gamma.o: sample_conv.ho
sample_conv_gamma.o: sample_conv_gamma.ho

sign_get.o: sign_get.c
sign_get.o: sign_get.ho
sign_get.o: sign.ho

vec.o: vec.c
vec.o: vec.ho
vec.o: affirm.ho

wt_table_binomial.o: wt_table_binomial.c
wt_table_binomial.o: affirm.ho
wt_table_binomial.o: gauss_distr.ho
wt_table_binomial.o: wt_table_binomial.ho

wt_table.o: wt_table.c
wt_table.o: affirm.ho
wt_table.o: jsstring.ho
wt_table.o: jsprintf.ho
wt_table.o: wt_table.ho

wt_table_hann.o: wt_table_hann.c
wt_table_hann.o: affirm.ho
wt_table_hann.o: gauss_distr.ho
wt_table_hann.o: wt_table_hann.ho

