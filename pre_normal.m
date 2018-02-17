function norm_01= pre_normal(data, dim) %normal data to [0 1]
if nargin < 2, dim = 2; end
min_col = min(data, [ ], dim);
max_col = max(data, [ ], dim);

norm_01 = bsxfun(@times, bsxfun(@minus, data, min_col), 1./(max_col - min_col)); %y=(x-min)/(max-min)

    