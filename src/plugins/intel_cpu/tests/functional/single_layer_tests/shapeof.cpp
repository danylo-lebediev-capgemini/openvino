// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/builders.hpp"
#include "ngraph_functions/utils/ngraph_helpers.hpp"
#include "shared_test_classes/base/ov_subgraph.hpp"
#include "test_utils/cpu_test_utils.hpp"

using namespace ov::test;
using namespace InferenceEngine;
using namespace CPUTestUtils;

using ElementType = ov::element::Type_t;

namespace CPULayerTestsDefinitions {
typedef std::tuple<
        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>  // input shape
> ShapeOfSpecificParams;

typedef std::tuple<
        ShapeOfSpecificParams,
        ElementType                    // Net precision
> ShapeOfLayerTestParams;

typedef std::tuple<
        CPULayerTestsDefinitions::ShapeOfLayerTestParams,
        CPUSpecificParams> ShapeOfLayerCPUTestParamsSet;

class ShapeOfLayerCPUTest : public testing::WithParamInterface<ShapeOfLayerCPUTestParamsSet>,
                            virtual public ov::test::SubgraphBaseTest, public CPUTestsBase {
public:
    static std::string getTestCaseName(testing::TestParamInfo<ShapeOfLayerCPUTestParamsSet> obj) {
        CPULayerTestsDefinitions::ShapeOfLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = obj.param;
        ElementType netPr;
        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;

        ShapeOfSpecificParams shapeOfPar;
        std::tie(shapeOfPar, netPr) = basicParamsSet;
        std::tie(shapes) = shapeOfPar;
        std::ostringstream result;
        result << "ShapeOfTest_";
        result << std::to_string(obj.index) << "_";
        result << "Prec=" << netPr << "_";
        result << CPUTestsBase::getTestCaseName(cpuParams) << "_";
        result << "IS=";
        for (const auto& shape : shapes.second) {
            result << "(";
            for (const auto& item : shape) {
                result << CommonTestUtils::vec2str(item);
            }
            result << ")_";
        }
        return result.str();
    }
protected:
    void SetUp() override {
        targetDevice = CommonTestUtils::DEVICE_CPU;

        CPULayerTestsDefinitions::ShapeOfLayerTestParams basicParamsSet;
        CPUSpecificParams cpuParams;
        std::tie(basicParamsSet, cpuParams) = this->GetParam();
        std::tie(inFmts, outFmts, priority, selectedType) = cpuParams;

        CPULayerTestsDefinitions::ShapeOfSpecificParams shapeOfParams;
        auto netPrecision = ElementType::undefined;
        std::tie(shapeOfParams, netPrecision) = basicParamsSet;

        std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>> shapes;
        std::tie(shapes) = shapeOfParams;
        inputDynamicShapes = shapes.first;
        targetStaticShapes = shapes.second;

        inType = netPrecision;
        outType = ngraph::element::i32;

        selectedType = makeSelectedTypeStr(getPrimitiveType(), netPrecision);

        //init_input_shapes(shapes);

        //auto params = ngraph::builder::makeParams(ov::element::Type(netPrecision), {targetStaticShapes.front().front()});
        auto params = ngraph::builder::makeDynamicParams(ov::element::Type(netPrecision), {targetStaticShapes.front().front()});
        auto paramOuts = ngraph::helpers::convert2OutputVector(ngraph::helpers::castOps2Nodes<ngraph::opset3::Parameter>(params));
        auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(paramOuts[0], outType);
        shapeOf->get_rt_info() = getCPUInfo();

        function = makeNgraphFunction(netPrecision, params, shapeOf, "ShapeOf");
        functionRefs = ngraph::clone_function(*function);
    }
};

TEST_P(ShapeOfLayerCPUTest, CompareWithRefs) {
    SKIP_IF_CURRENT_TEST_IS_DISABLED()
    run();
    CheckPluginRelatedResults(compiledModel, "ShapeOf");
}

namespace {

/* CPU PARAMS */
std::vector<CPUSpecificParams> filterCPUInfoForDevice(const size_t dimsCount = 3) {
    std::vector<CPUSpecificParams> resCPUParams;
    if (dimsCount == 5) {
        resCPUParams.push_back(CPUSpecificParams{{nCdhw16c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nCdhw8c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{ncdhw}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{ndhwc}, {x}, {}, {}});
    } else if (dimsCount == 4) {
        resCPUParams.push_back(CPUSpecificParams{{nChw16c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nChw8c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nchw}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nhwc}, {x}, {}, {}});
    } else {
        resCPUParams.push_back(CPUSpecificParams{{nCw16c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{nCw8c}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{abc}, {x}, {}, {}});
        resCPUParams.push_back(CPUSpecificParams{{acb}, {x}, {}, {}});
    }

    return resCPUParams;
}

const std::vector<ElementType> netPrecisions = {
        ElementType::f32,
        ElementType::bf16,
        ElementType::i32,
        ElementType::i8
};

std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic3d = {
        {{ngraph::PartialShape{-1, -1, -1}},
                {{{ 8, 5, 4 }, { 8, 5, 3 }, { 8, 5, 2 }}}},
        {{ngraph::PartialShape{-1, -1, -1}},
                {{{ 1, 2, 4 }, { 1, 2, 3 }, { 1, 2, 2 }}}}
};
std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic4d = {
        {{ngraph::PartialShape{-1, -1, -1, -1}},
                {{{ 8, 5, 3, 4 }, { 8, 5, 3, 3 }, { 8, 5, 3, 2 }}}},
        {{ngraph::PartialShape{-1, -1, -1, -1}},
                {{{ 1, 2, 3, 4 }, { 1, 2, 3, 3 }, { 1, 2, 3, 2 }}}}
};
std::vector<std::pair<std::vector<ngraph::PartialShape>, std::vector<std::vector<ngraph::Shape>>>> inShapesDynamic5d = {
        {{ngraph::PartialShape{-1, -1, -1, -1, -1}},
         {{{ 8, 5, 3, 2, 4 }, { 8, 5, 3, 2, 3 }, { 8, 5, 3, 2, 2 }}}},
         {{ngraph::PartialShape{-1, -1, -1, -1, -1}},
          {{{ 1, 2, 3, 4, 4 }, { 1, 2, 3, 4, 3 }, { 1, 2, 3, 4, 2 }}}}
};

const auto params3dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(inShapesDynamic3d)),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(filterCPUInfoForDevice(3)));

const auto params4dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(inShapesDynamic4d)),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(filterCPUInfoForDevice(4)));

const auto params5dDynamic = ::testing::Combine(
        ::testing::Combine(
                ::testing::Combine(
                        ::testing::ValuesIn(inShapesDynamic5d)),
                ::testing::ValuesIn(netPrecisions)),
        ::testing::ValuesIn(filterCPUInfoForDevice(5)));

// We don't check static case, because of constant folding
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf3dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params3dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf4dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params4dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
INSTANTIATE_TEST_SUITE_P(smoke_ShapeOf5dDynamicLayoutTest, ShapeOfLayerCPUTest,
                         params5dDynamic, ShapeOfLayerCPUTest::getTestCaseName);
} // namespace
} // namespace CPULayerTestsDefinitions
