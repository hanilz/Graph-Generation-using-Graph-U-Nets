function [P R W c] = AMG(fW, beta, NS)
%
% Graph coarsening using Algebraic MultiGrid (AMG)
%
% Usage: 
%   [P R W c] = AMG(fW, beta, NS)
%
% Inputs:
%   fW      - affinity matrix at the finest scale. 
%             Matrix must be symmetric, sparse and have non-negative
%             entries.
%   beta    - coarsening factor (typical value ~.2)
%   NS      - number of scales / levels
%
% Outputs
%   P       - interp matrices (from coarse to fine)
%   R       - restriction matrices (from fine to coarse)
%   W       - affective affinity matrix at each scale (W{1}=fW)
%   c       - coarse nodes selected at each scale
%
% Example:
%   Image super-pixels
%
%   img = im2double(imread('football.jpg'));
%   sz = size(img);
%   % forming 4-connected graph over image grid
%   [ii jj] = sparse_adj_matrix(sz(1:2), 1, 1, 1);
%   fimg = reshape(img,[],sz(3));
%   wij = fimg(ii,:)-fimg(jj,:);
%   wij = sum(wij.^2, 2);
%   wij = exp(-wij./(2*mean(wij)));
%   fW = sparse(ii, jj, wij, prod(sz(1:2)), prod(sz(1:2)));
%
%   % forming AMG
%   [P R W c] = AMG(fW, .2, 5);
%
%   % super pixels
%   [~, sp] = max( P{1}*(P{2}*(P{3}*(P{4}*P{5}))), [], 2);
%
%   figure;
%   subplot(121);imshow(img);title('input image');
%   subplot(122);imagesc(reshape(sp,sz(1:2)));
%   axis image;colormap(rand(numel(c{5}),3));
%   title('super pixels');
%
%
%  Copyright (c) Bagon Shai
%  Many thanks to Meirav Galun.
%  Department of Computer Science and Applied Mathmatics
%  Wiezmann Institute of Science
%  http://www.wisdom.weizmann.ac.il/
% 
%  Permission is hereby granted, free of charge, to any person obtaining a copy
%  of this software and associated documentation files (the "Software"), to deal
%  in the Software without restriction, subject to the following conditions:
% 
%  1. The above copyright notice and this permission notice shall be included in
%      all copies or substantial portions of the Software.
%  2. No commercial use will be done with this software.
%  3. If used in an academic framework - a proper citation must be included.
%
% @ELECTRONIC{bagon2012,
%  author = {Shai Bagon},
%  title = {Matlab implementation for AMG graph coarsening},
%  url = {http://www.wisdom.weizmann.ac.il/~bagon/matlab.html#AMG},
%  owner = {bagon},
%  version = {}, <-please insert version number from VERSION.txt
% }
%
% 
%  The Software is provided "as is", without warranty of any kind.
% 
%  May 2011
% 


n = size(fW,1);

P = cell(1,NS);
c = cell(1,NS);
W = cell(1,NS);

% make sure diagonal is zero
W{1} = fW - spdiags( spdiags(fW,0), 0, n, n);

fine = 1:n;

for si=1:NS
    
    [tmp_c P{si}] = fine2coarse(W{si}, beta);    
    c{si} = fine(tmp_c);
    
    if si<NS
        
        W{si+1} = P{si}'*W{si}*P{si};
        
        W{si+1} = W{si+1} - spdiags( spdiags(W{si+1},0), 0, size(W{si+1},1), size(W{si+1},2));
        fine = c{si};
    end
end
% restriction matrices
R = cellfun(@(x) spmtimesd(x', 1./sum(x,1), []), P, 'UniformOutput', false);


%-------------------------------------------------------------------------%
function [c P] = fine2coarse(W, beta)
%
% Performs one step of coarsening
%

n = size(W,1);
% weight normalization
nW = spmtimesd(W, 1./full(sum(W,2)), []);

% % % select coarse nodes (mex implementation)
% c = ChooseCoarseGreedy_mex(nW, randperm(n), beta);

c = false(1,n);
sum_jc = zeros(n,1);
ord = [202, 433, 102, 76, 384, 546, 301, 601, 362, 262, 339, 29, 615, 377, 22, 508, 290, 190, 394, 537, 437, 458, 356, 572, 88, 563, 517, 103, 126, 8, 74, 35, 470, 570, 53, 195, 329, 114, 86, 12, 313, 382, 578, 72, 630, 127, 235, 224, 93, 2, 418, 547, 146, 154, 379, 353, 128, 92, 168, 201, 79, 560, 278, 211, 457, 453, 50, 164, 468, 82, 62, 119, 170, 3, 274, 395, 451, 207, 97, 181, 134, 171, 135, 40, 293, 558, 552, 659, 485, 261, 199, 428, 209, 120, 200, 132, 600, 21, 69, 210, 477, 525, 391, 527, 594, 212, 4, 338, 649, 162, 556, 286, 241, 243, 305, 281, 229, 5, 213, 232, 280, 56, 347, 36, 447, 19, 51, 462, 355, 299, 393, 452, 341, 66, 222, 183, 539, 455, 573, 214, 78, 652, 156, 153, 628, 57, 464, 223, 277, 70, 217, 427, 392, 351, 334, 441, 374, 312, 159, 512, 550, 142, 446, 166, 269, 520, 73, 407, 643, 372, 218, 300, 310, 233, 554, 349, 184, 553, 442, 589, 189, 371, 661, 511, 523, 663, 656, 581, 606, 45, 110, 263, 173, 100, 165, 248, 408, 25, 85, 87, 253, 68, 603, 24, 571, 80, 593, 429, 461, 228, 196, 216, 321, 215, 264, 492, 295, 504, 137, 496, 38, 555, 84, 26, 252, 55, 405, 116, 99, 463, 564, 483, 423, 358, 33, 557, 518, 246, 534, 398, 421, 332, 509, 256, 52, 498, 543, 596, 529, 147, 152, 148, 343, 619, 623, 352, 438, 242, 367, 360, 337, 157, 641, 478, 666, 319, 469, 188, 113, 234, 489, 20, 297, 562, 239, 237, 551, 169, 631, 660, 399, 411, 426, 500, 46, 298, 637, 318, 340, 255, 203, 259, 260, 192, 472, 176, 179, 586, 309, 75, 328, 34, 616, 535, 651, 519, 373, 193, 254, 425, 642, 81, 268, 276, 47, 27, 60, 587, 595, 444, 161, 635, 122, 475, 186, 532, 454, 501, 667, 471, 531, 591, 89, 325, 465, 417, 18, 284, 618, 54, 294, 106, 410, 370, 249, 167, 206, 96, 387, 378, 42, 288, 582, 567, 296, 612, 495, 121, 158, 415, 493, 221, 112, 459, 227, 647, 257, 31, 602, 588, 65, 346, 7, 369, 614, 308, 577, 140, 414, 416, 541, 528, 515, 624, 67, 251, 609, 150, 648, 412, 105, 174, 336, 117, 670, 194, 396, 368, 482, 497, 544, 240, 6, 230, 445, 514, 390, 361, 467, 91, 331, 208, 326, 354, 136, 160, 317, 617, 487, 592, 28, 549, 335, 265, 303, 138, 629, 180, 204, 516, 359, 406, 510, 37, 658, 282, 632, 540, 104, 627, 599, 315, 443, 657, 191, 542, 59, 402, 289, 409, 275, 131, 385, 236, 149, 247, 568, 111, 432, 502, 177, 668, 198, 285, 220, 439, 226, 43, 108, 513, 481, 521, 316, 584, 662, 419, 574, 575, 133, 205, 357, 342, 430, 613, 10, 440, 107, 486, 524, 345, 125, 143, 565, 64, 11, 506, 141, 272, 118, 258, 273, 63, 365, 101, 292, 590, 139, 490, 413, 311, 499, 636, 178, 244, 271, 90, 654, 270, 48, 404, 449, 314, 187, 436, 155, 460, 197, 94, 115, 598, 109, 163, 431, 386, 644, 172, 17, 597, 323, 151, 307, 566, 185, 129, 124, 580, 364, 607, 380, 375, 530, 291, 634, 545, 474, 366, 322, 320, 49, 279, 98, 665, 621, 448, 250, 488, 669, 491, 526, 505, 585, 344, 30, 434, 622, 324, 32, 522, 16, 13, 640, 330, 145, 245, 625, 533, 576, 655, 645, 610, 559, 388, 424, 123, 219, 569, 283, 476, 71, 456, 608, 664, 77, 39, 389, 653, 650, 95, 58, 466, 363, 61, 646, 225, 480, 306, 435, 579, 302, 350, 479, 561, 9, 304, 626, 633, 182, 14, 450, 267, 484, 620, 23, 383, 604, 397, 130, 611, 401, 1, 327, 287, 507, 266, 348, 536, 175, 144, 422, 638, 503, 538, 548, 494, 605, 238, 400, 15, 583, 376, 231, 639, 403, 41, 381, 83, 420, 333, 473, 44];
% % for ii=1:n % lexicographic order - any better ideas?
% for ii = randperm(n) % random order    
for ii = ord % random order    
    if sum_jc(ii) <= beta
        c(ii)=true;
        sum_jc = sum_jc + nW(:,ii); 
    end
end


% compute the interp matrix
ci=find(c);
P = W(:,ci);
P = spmtimesd(P, 1./full(sum(P,2)), []);
% make sure coarse points are directly connected to their fine counterparts
[jj ii pji] = find(P'); 
sel = ~c(ii); % select only fine points
mycat = @(x,y) vertcat(x(:),y(:));
P = sparse( mycat(jj(sel), 1:sum(c)),...
    mycat(ii(sel), ci),...
    mycat(pji(sel), ones(1,sum(c))), size(P,2), size(P,1))';
